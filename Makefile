# Streaming Integration Test
#
#   make up     Provision Pub/Sub + VM, wait for poller, start pipeline
#   make down   Stop pipeline, delete VM + Pub/Sub + GCS staging
#   make logs   Tail the pipeline log (target station events)

PROJECT  ?= realtime-headway-prediction
REGION   ?= us-east1
ZONE     ?= us-east1-b
VM       ?= gtfs-poller
SA       ?= mlops-sa@$(PROJECT).iam.gserviceaccount.com
TOPIC    ?= gtfs-rt-ace
SUB      ?= gtfs-rt-ace-sub
SUB_FULL ?= projects/$(PROJECT)/subscriptions/$(SUB)
BUCKET   ?= gs://$(PROJECT)-poller-staging
PID_FILE := .pipeline.pid
LOG_FILE := streaming-pipeline.log

# Production Dataflow settings
DATAFLOW_BUCKET   ?= gs://$(PROJECT)-dataflow-staging
DATAFLOW_JOB_NAME ?= headway-streaming
DATAFLOW_WORKER   ?= n1-standard-2
DATAFLOW_MAX_WORKERS ?= 3

# ──────────────────────────────────────────────────────────────
# Integration Test (local DirectRunner)
# ──────────────────────────────────────────────────────────────
.PHONY: up down logs help

up:
	@echo ""
	@echo "=== STARTING INTEGRATION TEST ==="
	@echo ""
	@echo "[1/5] Pub/Sub topic + subscription..."
	@gcloud pubsub topics describe $(TOPIC) --project=$(PROJECT) >/dev/null 2>&1 \
		|| gcloud pubsub topics create $(TOPIC) --project=$(PROJECT)
	@gcloud pubsub subscriptions describe $(SUB) --project=$(PROJECT) >/dev/null 2>&1 \
		|| gcloud pubsub subscriptions create $(SUB) \
			--topic=$(TOPIC) --project=$(PROJECT) \
			--ack-deadline=60 --message-retention-duration=1h
	@echo "  $(TOPIC) / $(SUB)"
	@echo ""
	@echo "[2/5] Upload poller to GCS..."
	@gsutil ls $(BUCKET) >/dev/null 2>&1 || gsutil mb -l $(REGION) -p $(PROJECT) $(BUCKET)
	@tar -czf /tmp/_poller.tar.gz -C realtime_ingestion poller/
	@gsutil -q cp /tmp/_poller.tar.gz $(BUCKET)/poller-code.tar.gz
	@rm -f /tmp/_poller.tar.gz
	@echo "  $(BUCKET)/poller-code.tar.gz"
	@echo ""
	@echo "[3/5] Create VM (startup script installs + starts poller)..."
	@if gcloud compute instances describe $(VM) --zone=$(ZONE) --project=$(PROJECT) >/dev/null 2>&1; then \
		echo "  VM already exists"; \
	else \
		gcloud compute instances create $(VM) \
			--project=$(PROJECT) --zone=$(ZONE) \
			--machine-type=e2-small \
			--service-account=$(SA) --scopes=cloud-platform \
			--image-family=debian-12 --image-project=debian-cloud \
			--metadata-from-file=startup-script=infra/poller_startup.sh \
			--tags=gtfs-poller --quiet; \
		echo "  Created $(VM)"; \
	fi
	@echo ""
	@echo "[4/5] Waiting for poller to be ready..."
	@for i in $$(seq 1 24); do \
		if gcloud compute ssh $(VM) --zone=$(ZONE) --project=$(PROJECT) \
			--quiet --strict-host-key-checking=no \
			--command="grep -q 'startup script complete' /var/log/poller-setup.log 2>/dev/null && echo READY" 2>/dev/null \
			| grep -q READY; then \
			echo "  Poller ready."; \
			break; \
		fi; \
		if [ $$i -eq 24 ]; then echo "  Timed out (2 min). Debug: gcloud compute ssh $(VM) --zone=$(ZONE) --command='cat /var/log/poller-setup.log'"; exit 1; fi; \
		sleep 5; \
	done
	@echo ""
	@echo "[5/5] Starting streaming pipeline..."
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "  Pipeline already running (PID $$(cat $(PID_FILE)))"; \
	else \
		python3 -m pipelines.beam.streaming.streaming_pipeline \
			--input_subscription $(SUB_FULL) --project_id $(PROJECT) \
			>$(LOG_FILE) 2>&1 & \
		echo $$! > $(PID_FILE); \
		echo "  Pipeline started (PID $$(cat $(PID_FILE)))"; \
	fi
	@echo ""
	@echo "=== RUNNING — use 'make logs' to watch target station events ==="

down:
	@echo ""
	@echo "=== TEARING DOWN ==="
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		kill $$PID 2>/dev/null; sleep 1; kill -9 $$PID 2>/dev/null; \
		rm -f $(PID_FILE); echo "  Pipeline stopped"; \
	fi
	@rm -f $(LOG_FILE)
	-@gcloud compute instances delete $(VM) --zone=$(ZONE) --project=$(PROJECT) --quiet 2>/dev/null && echo "  VM deleted" || true
	-@gcloud pubsub subscriptions delete $(SUB) --project=$(PROJECT) --quiet 2>/dev/null && echo "  Subscription deleted" || true
	-@gcloud pubsub topics delete $(TOPIC) --project=$(PROJECT) --quiet 2>/dev/null && echo "  Topic deleted" || true
	-@gsutil rm -r $(BUCKET) 2>/dev/null && echo "  GCS staging deleted" || true
	@echo "=== DONE ==="

logs:
	@tail -f $(LOG_FILE)

# ──────────────────────────────────────────────────────────────
# Production (Dataflow runner)
# ──────────────────────────────────────────────────────────────
.PHONY: deploy-infra start-ingestion pause-ingestion restart-ingestion teardown prod-status prod-logs

deploy-infra:
	@echo ""
	@echo "=== DEPLOYING PRODUCTION INFRASTRUCTURE ==="
	@echo ""
	@echo "[1/4] Pub/Sub topic..."
	@gcloud pubsub topics describe $(TOPIC) --project=$(PROJECT) >/dev/null 2>&1 \
		|| gcloud pubsub topics create $(TOPIC) --project=$(PROJECT)
	@echo "  Topic: $(TOPIC)"
	@echo ""
	@echo "[2/4] Pub/Sub subscription (7-day retention for Dataflow)..."
	@gcloud pubsub subscriptions describe $(SUB) --project=$(PROJECT) >/dev/null 2>&1 \
		|| gcloud pubsub subscriptions create $(SUB) \
			--topic=$(TOPIC) --project=$(PROJECT) \
			--ack-deadline=120 \
			--message-retention-duration=7d \
			--expiration-period=never
	@echo "  Subscription: $(SUB)"
	@echo ""
	@echo "[3/4] Upload poller code to GCS..."
	@gsutil ls $(BUCKET) >/dev/null 2>&1 || gsutil mb -l $(REGION) -p $(PROJECT) $(BUCKET)
	@tar -czf /tmp/_poller.tar.gz -C realtime_ingestion poller/
	@gsutil -q cp /tmp/_poller.tar.gz $(BUCKET)/poller-code.tar.gz
	@rm -f /tmp/_poller.tar.gz
	@echo "  $(BUCKET)/poller-code.tar.gz"
	@echo ""
	@echo "[4/4] Create poller VM..."
	@if gcloud compute instances describe $(VM) --zone=$(ZONE) --project=$(PROJECT) >/dev/null 2>&1; then \
		echo "  VM already exists — skipping"; \
	else \
		gcloud compute instances create $(VM) \
			--project=$(PROJECT) --zone=$(ZONE) \
			--machine-type=e2-small \
			--service-account=$(SA) --scopes=cloud-platform \
			--image-family=debian-12 --image-project=debian-cloud \
			--metadata-from-file=startup-script=infra/poller_startup.sh \
			--tags=gtfs-poller --quiet; \
		echo "  Created $(VM)"; \
	fi
	@echo ""
	@echo "[4/4] Dataflow staging bucket..."
	@gsutil ls $(DATAFLOW_BUCKET) >/dev/null 2>&1 \
		|| gsutil mb -l $(REGION) -p $(PROJECT) $(DATAFLOW_BUCKET)
	@echo "  $(DATAFLOW_BUCKET)"
	@echo ""
	@echo "=== INFRASTRUCTURE READY ==="
	@echo ""
	@echo "Next: make start-ingestion"

start-ingestion:
	@echo ""
	@echo "=== STARTING PRODUCTION INGESTION ==="
	@echo ""
	@echo "[1/3] Waiting for poller VM to be ready..."
	@for i in $$(seq 1 36); do \
		if gcloud compute ssh $(VM) --zone=$(ZONE) --project=$(PROJECT) \
			--quiet --strict-host-key-checking=no \
			--command="grep -q 'startup script complete' /var/log/poller-setup.log 2>/dev/null && echo READY" 2>/dev/null \
			| grep -q READY; then \
			echo "  Poller ready."; \
			break; \
		fi; \
		if [ $$i -eq 36 ]; then echo "  Timed out (3 min)."; echo "  Debug: gcloud compute ssh $(VM) --zone=$(ZONE) --command='cat /var/log/poller-setup.log'"; exit 1; fi; \
		sleep 5; \
	done
	@echo ""
	@echo "[2/3] Verifying poller is publishing..."
	@sleep 10
	@MSG_COUNT=$$(gcloud pubsub subscriptions pull $(SUB) --project=$(PROJECT) \
		--limit=1 --auto-ack --format='value(message.data)' 2>/dev/null | wc -c | tr -d ' '); \
	if [ "$$MSG_COUNT" -gt "10" ]; then \
		echo "  Poller publishing — messages detected on $(SUB)"; \
	else \
		echo "  WARNING: No messages detected yet. Poller may still be starting."; \
		echo "  Check: gcloud compute ssh $(VM) --zone=$(ZONE) --command='sudo journalctl -u gtfs-poller -n 20'"; \
	fi
	@echo ""
	@echo "[3/3] Launching Dataflow streaming pipeline..."
	@python3 -m pipelines.beam.streaming.streaming_pipeline \
		--input_subscription $(SUB_FULL) \
		--project_id $(PROJECT) \
		--region $(REGION) \
		--runner DataflowRunner \
		--project $(PROJECT) \
		--region $(REGION) \
		--staging_location $(DATAFLOW_BUCKET)/staging \
		--temp_location $(DATAFLOW_BUCKET)/temp \
		--setup_file ./setup.py \
		--job_name $(DATAFLOW_JOB_NAME) \
		--machine_type $(DATAFLOW_WORKER) \
		--max_num_workers $(DATAFLOW_MAX_WORKERS) \
		--streaming \
		--save_main_session \
		--service_account_email $(SA) \
		--enable_streaming_engine
	@echo ""
	@echo "=== INGESTION RUNNING ==="
	@echo ""
	@echo "  Poller VM:  $(VM) → Pub/Sub $(TOPIC)"
	@echo "  Dataflow:   $(DATAFLOW_JOB_NAME)"
	@echo "  Buffer:     ~10-15 min warmup (20 observations per group)"
	@echo ""
	@echo "  Monitor:    make prod-status"
	@echo "  Logs:       make prod-logs"
	@echo "  Console:    https://console.cloud.google.com/dataflow/jobs/$(REGION)?project=$(PROJECT)"

pause-ingestion:
	@echo ""
	@echo "=== PAUSING POLLER ==="
	@echo ""
	@echo "Stopping poller on VM..."
	@gcloud compute ssh $(VM) --zone=$(ZONE) --project=$(PROJECT) \
		--quiet --strict-host-key-checking=no \
		--command="sudo systemctl stop gtfs-poller" 2>/dev/null \
		&& echo "  Poller stopped" \
		|| echo "  WARNING: Could not stop poller (VM may not be running)"
	@echo ""
	@echo "=== POLLER PAUSED ==="
	@echo "  Dataflow is still running (will idle until new messages arrive)."
	@echo "  Resume: make restart-ingestion"

restart-ingestion:
	@echo ""
	@echo "=== RESTARTING POLLER ==="
	@echo ""
	@echo "Starting poller on VM..."
	@gcloud compute ssh $(VM) --zone=$(ZONE) --project=$(PROJECT) \
		--quiet --strict-host-key-checking=no \
		--command="sudo systemctl start gtfs-poller" 2>/dev/null \
		&& echo "  Poller started" \
		|| echo "  WARNING: Could not start poller (is VM running?)"
	@echo ""
	@echo "=== POLLER RESTARTED ==="

teardown:
	@echo ""
	@echo "=== TEARING DOWN PRODUCTION INFRASTRUCTURE ==="
	@echo ""
	@echo "[1/4] Cancelling Dataflow job..."
	@JOB_ID=$$(gcloud dataflow jobs list \
		--project=$(PROJECT) --region=$(REGION) \
		--filter="name=$(DATAFLOW_JOB_NAME) AND (state=Running OR state=Draining)" \
		--format="value(JOB_ID)" --limit=1 2>/dev/null); \
	if [ -n "$$JOB_ID" ]; then \
		gcloud dataflow jobs cancel "$$JOB_ID" \
			--project=$(PROJECT) --region=$(REGION) --quiet; \
		echo "  Dataflow job $$JOB_ID cancelled"; \
	else \
		echo "  No active Dataflow job found"; \
	fi
	@echo ""
	@echo "[2/4] Deleting poller VM..."
	-@gcloud compute instances delete $(VM) --zone=$(ZONE) --project=$(PROJECT) --quiet 2>/dev/null \
		&& echo "  VM deleted" || echo "  VM not found"
	@echo ""
	@echo "[3/4] Deleting Pub/Sub resources..."
	-@gcloud pubsub subscriptions delete $(SUB) --project=$(PROJECT) --quiet 2>/dev/null \
		&& echo "  Subscription deleted" || echo "  Subscription not found"
	-@gcloud pubsub topics delete $(TOPIC) --project=$(PROJECT) --quiet 2>/dev/null \
		&& echo "  Topic deleted" || echo "  Topic not found"
	@echo ""
	@echo "[4/4] Cleaning GCS staging..."
	-@gsutil rm -r $(BUCKET) 2>/dev/null && echo "  Poller staging deleted" || true
	-@gsutil rm -r $(DATAFLOW_BUCKET) 2>/dev/null && echo "  Dataflow staging deleted" || true
	@echo ""
	@echo "=== TEARDOWN COMPLETE ==="
	@echo ""
	@echo "  NOTE: Prediction endpoint NOT deleted (managed separately)."
	@echo "  To delete: python scripts/deploy_endpoint.py --undeploy"

prod-status:
	@echo ""
	@echo "=== PRODUCTION STATUS ==="
	@echo ""
	@echo "--- Poller VM ---"
	@gcloud compute instances describe $(VM) --zone=$(ZONE) --project=$(PROJECT) \
		--format="table(name, status, networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null \
		|| echo "  VM not found"
	@echo ""
	@gcloud compute ssh $(VM) --zone=$(ZONE) --project=$(PROJECT) \
		--quiet --strict-host-key-checking=no \
		--command="sudo systemctl is-active gtfs-poller 2>/dev/null && echo '  Poller: RUNNING' || echo '  Poller: STOPPED'" 2>/dev/null \
		|| echo "  Cannot reach VM"
	@echo ""
	@echo "--- Pub/Sub ---"
	@gcloud pubsub subscriptions describe $(SUB) --project=$(PROJECT) \
		--format="value(pushConfig, messageRetentionDuration)" 2>/dev/null \
		|| echo "  Subscription not found"
	@echo ""
	@echo "--- Dataflow ---"
	@gcloud dataflow jobs list \
		--project=$(PROJECT) --region=$(REGION) \
		--filter="name=$(DATAFLOW_JOB_NAME)" \
		--format="table(JOB_ID, NAME, STATE, START_TIME)" \
		--limit=3 2>/dev/null \
		|| echo "  No Dataflow jobs found"
	@echo ""

prod-logs:
	@echo "Fetching recent Dataflow logs..."
	@JOB_ID=$$(gcloud dataflow jobs list \
		--project=$(PROJECT) --region=$(REGION) \
		--filter="name=$(DATAFLOW_JOB_NAME) AND state=Running" \
		--format="value(JOB_ID)" --limit=1 2>/dev/null); \
	if [ -n "$$JOB_ID" ]; then \
		gcloud logging read "resource.type=dataflow_step AND resource.labels.job_id=$$JOB_ID" \
			--project=$(PROJECT) --limit=50 --format="table(timestamp, textPayload)" \
			--freshness=30m --order=asc; \
	else \
		echo "No running Dataflow job found. Showing poller logs instead:"; \
		gcloud compute ssh $(VM) --zone=$(ZONE) --project=$(PROJECT) \
			--quiet --strict-host-key-checking=no \
			--command="sudo journalctl -u gtfs-poller -n 30 --no-pager" 2>/dev/null \
			|| echo "Cannot reach poller VM"; \
	fi

help:
	@echo ""
	@echo "  Integration Test (local DirectRunner):"
	@echo "    make up     Provision infra, start poller + local pipeline"
	@echo "    make down   Stop everything, delete all resources"
	@echo "    make logs   Tail local pipeline log"
	@echo ""
	@echo "  Production (Dataflow):"
	@echo "    make deploy-infra        Provision VM + Pub/Sub + GCS staging"
	@echo "    make start-ingestion     Start poller + launch Dataflow pipeline"
	@echo "    make pause-ingestion     Stop poller (Dataflow keeps running)"
	@echo "    make restart-ingestion   Resume poller"
	@echo "    make teardown            Delete all production resources"
	@echo "    make prod-status         Check status of all components"
	@echo "    make prod-logs           Tail Dataflow or poller logs"
	@echo ""
