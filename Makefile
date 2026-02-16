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
		python -m pipelines.beam.streaming.streaming_pipeline \
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

help:
	@echo "  make up     Provision infra, start poller + pipeline"
	@echo "  make down   Stop everything, delete all resources"
	@echo "  make logs   Tail pipeline log (target station events)"

.DEFAULT_GOAL := help
