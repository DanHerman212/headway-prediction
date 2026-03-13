# Graph WaveNet Dense-Grid Dataset Pipeline
#
#   make gw-infra-up          Provision VM + Pub/Sub + BigQuery
#   make gw-start-ingestion   Start poller + Dataflow
#   make gw-start-dataflow    Launch Dataflow only
#   make gw-pause             Pause poller
#   make gw-resume            Resume poller
#   make gw-status            Check all components
#   make gw-logs              Tail logs
#   make gw-infra-down        Tear down everything

PROJECT  ?= realtime-headway-prediction
REGION   ?= us-east1
ZONE     ?= us-east1-b
SA       ?= mlops-sa@$(PROJECT).iam.gserviceaccount.com

GW_VM        ?= gw-grid-publisher
GW_TOPIC     ?= graph-wavenet-snapshots
GW_SUB       ?= graph-wavenet-snapshots-dataflow
GW_SUB_FULL  ?= projects/$(PROJECT)/subscriptions/$(GW_SUB)
GW_BUCKET    ?= gs://$(PROJECT)-gw-staging
GW_BQ_DS     ?= graph_wavenet
GW_BQ_TABLE  ?= dense_grid_labeled
GW_BQ_FULL   ?= $(PROJECT):$(GW_BQ_DS).$(GW_BQ_TABLE)
GW_DF_JOB    ?= gw-dense-grid-labeling
GW_DF_BUCKET ?= gs://$(PROJECT)-gw-dataflow
GW_DF_WORKER ?= n1-standard-2
GW_DF_MAX    ?= 2
GW_SERVICE   ?= gw-grid-publisher

.PHONY: gw-infra-up gw-start-ingestion gw-start-dataflow gw-pause gw-resume gw-status gw-logs gw-infra-down

# ------------------------------------------------------------------
# 1) Provision all infrastructure
# ------------------------------------------------------------------
gw-infra-up:
	@echo ""
	@echo "=== GRAPH WAVENET: PROVISIONING INFRASTRUCTURE ==="
	@echo ""
	@echo "[1/5] Pub/Sub topic..."
	@gcloud pubsub topics describe $(GW_TOPIC) --project=$(PROJECT) >/dev/null 2>&1 \
		|| gcloud pubsub topics create $(GW_TOPIC) --project=$(PROJECT)
	@echo "  Topic: $(GW_TOPIC)"
	@echo ""
	@echo "[2/5] Pub/Sub subscription (7-day retention)..."
	@gcloud pubsub subscriptions describe $(GW_SUB) --project=$(PROJECT) >/dev/null 2>&1 \
		|| gcloud pubsub subscriptions create $(GW_SUB) \
			--topic=$(GW_TOPIC) --project=$(PROJECT) \
			--ack-deadline=120 \
			--message-retention-duration=7d \
			--expiration-period=never
	@echo "  Subscription: $(GW_SUB)"
	@echo ""
	@echo "[3/5] BigQuery dataset + table..."
	@bq --project_id=$(PROJECT) mk --force --dataset $(GW_BQ_DS)
	@bq --project_id=$(PROJECT) mk --force \
		--table \
		--time_partitioning_field snapshot_time \
		--time_partitioning_type DAY \
		--clustering_fields node_id \
		$(GW_BQ_DS).$(GW_BQ_TABLE) \
		snapshot_time:DATETIME,node_id:STRING,train_present:INTEGER,elapsed_headway:INTEGER,is_A:INTEGER,is_C:INTEGER,is_E:INTEGER,time_sin:FLOAT,time_cos:FLOAT,dow_sin:FLOAT,dow_cos:FLOAT,time_to_next_train:INTEGER
	@echo "  Table: $(GW_BQ_FULL)"
	@echo ""
	@echo "[4/5] Upload publisher code to GCS..."
	@gsutil ls $(GW_BUCKET) >/dev/null 2>&1 || gsutil mb -l $(REGION) -p $(PROJECT) $(GW_BUCKET)
	@rm -rf /tmp/_gw_publisher && mkdir -p /tmp/_gw_publisher
	@cp realtime_ingestion/graph_ingestion/grid_publisher.py /tmp/_gw_publisher/
	@cp realtime_ingestion/poller/gtfs_realtime_pb2.py /tmp/_gw_publisher/
	@cp realtime_ingestion/poller/nyct_subway_pb2.py /tmp/_gw_publisher/
	@cp local_artifacts/node_to_id.json /tmp/_gw_publisher/
	@tar -czf /tmp/gw-publisher.tar.gz -C /tmp/_gw_publisher .
	@gsutil -q cp /tmp/gw-publisher.tar.gz $(GW_BUCKET)/gw-publisher.tar.gz
	@rm -rf /tmp/_gw_publisher /tmp/gw-publisher.tar.gz
	@echo "  $(GW_BUCKET)/gw-publisher.tar.gz"
	@echo ""
	@echo "[5/5] Create poller VM..."
	@if gcloud compute instances describe $(GW_VM) --zone=$(ZONE) --project=$(PROJECT) >/dev/null 2>&1; then \
		echo "  VM already exists -- skipping"; \
	else \
		gcloud compute instances create $(GW_VM) \
			--project=$(PROJECT) --zone=$(ZONE) \
			--machine-type=e2-small \
			--service-account=$(SA) --scopes=cloud-platform \
			--image-family=debian-12 --image-project=debian-cloud \
			--metadata=gw-topic=$(GW_TOPIC) \
			--metadata-from-file=startup-script=infra/gw_poller_startup.sh \
			--tags=gw-publisher --quiet; \
		echo "  Created $(GW_VM)"; \
	fi
	@echo ""
	@echo "[5/5] Dataflow staging bucket..."
	@gsutil ls $(GW_DF_BUCKET) >/dev/null 2>&1 \
		|| gsutil mb -l $(REGION) -p $(PROJECT) $(GW_DF_BUCKET)
	@echo "  $(GW_DF_BUCKET)"
	@echo ""
	@echo "=== INFRASTRUCTURE READY ==="
	@echo ""
	@echo "Next: make gw-start-ingestion"

# ------------------------------------------------------------------
# 2) Start ingestion (poller + Dataflow)
# ------------------------------------------------------------------
gw-start-ingestion:
	@echo ""
	@echo "=== GRAPH WAVENET: STARTING INGESTION ==="
	@echo ""
	@echo "[1/3] Waiting for publisher VM to be ready..."
	@for i in $$(seq 1 36); do \
		if gcloud compute ssh $(GW_VM) --zone=$(ZONE) --project=$(PROJECT) \
			--quiet --strict-host-key-checking=no \
			--command="grep -q 'startup complete' /var/log/gw-poller-setup.log 2>/dev/null && echo READY" 2>/dev/null \
			| grep -q READY; then \
			echo "  Publisher ready."; \
			break; \
		fi; \
		if [ $$i -eq 36 ]; then \
			echo "  Timed out (3 min)."; \
			echo "  Debug: gcloud compute ssh $(GW_VM) --zone=$(ZONE) --command='cat /var/log/gw-poller-setup.log'"; \
			exit 1; \
		fi; \
		sleep 5; \
	done
	@echo ""
	@echo "[2/3] Verifying publisher is producing messages..."
	@sleep 15
	@MSG_COUNT=$$(gcloud pubsub subscriptions pull $(GW_SUB) --project=$(PROJECT) \
		--limit=1 --auto-ack --format='value(message.data)' 2>/dev/null | wc -c | tr -d ' '); \
	if [ "$$MSG_COUNT" -gt "10" ]; then \
		echo "  Publisher active -- messages on $(GW_SUB)"; \
	else \
		echo "  WARNING: No messages yet. Publisher may still be starting."; \
		echo "  Check: gcloud compute ssh $(GW_VM) --zone=$(ZONE) --command='sudo journalctl -u $(GW_SERVICE) -n 20'"; \
	fi
	@echo ""
	@$(MAKE) gw-start-dataflow
	@echo ""
	@echo "=== INGESTION RUNNING ==="
	@echo ""
	@echo "  Publisher: $(GW_VM) -> Pub/Sub $(GW_TOPIC)"
	@echo "  Dataflow:  $(GW_DF_JOB) -> BigQuery $(GW_BQ_FULL)"
	@echo ""
	@echo "  Monitor: make gw-status"
	@echo "  Logs:    make gw-logs"

# ------------------------------------------------------------------
# 3) Launch Dataflow only
# ------------------------------------------------------------------
gw-start-dataflow:
	@echo "[3/3] Launching Dataflow streaming pipeline..."
	@python3 -m pipelines.graph_wavenet.pipeline \
		--input_subscription $(GW_SUB_FULL) \
		--output_table $(GW_BQ_FULL) \
		--runner DataflowRunner \
		--project $(PROJECT) \
		--region $(REGION) \
		--staging_location $(GW_DF_BUCKET)/staging \
		--temp_location $(GW_DF_BUCKET)/temp \
		--setup_file ./setup.py \
		--job_name $(GW_DF_JOB) \
		--machine_type $(GW_DF_WORKER) \
		--max_num_workers $(GW_DF_MAX) \
		--streaming \
		--save_main_session \
		--service_account_email $(SA) \
		--enable_streaming_engine \
		--no_wait
	@echo "  Dataflow job submitted: $(GW_DF_JOB)"

# ------------------------------------------------------------------
# 4) Pause / Resume ingestion
# ------------------------------------------------------------------
gw-pause:
	@echo ""
	@echo "=== GRAPH WAVENET: PAUSING PUBLISHER ==="
	@gcloud compute ssh $(GW_VM) --zone=$(ZONE) --project=$(PROJECT) \
		--quiet --strict-host-key-checking=no \
		--command="sudo systemctl stop $(GW_SERVICE)" 2>/dev/null \
		&& echo "  Publisher stopped" \
		|| echo "  WARNING: Could not stop publisher (VM may not be running)"
	@echo "  Dataflow is still running (will idle until new messages)."
	@echo "  Resume: make gw-resume"

gw-resume:
	@echo ""
	@echo "=== GRAPH WAVENET: RESUMING PUBLISHER ==="
	@gcloud compute ssh $(GW_VM) --zone=$(ZONE) --project=$(PROJECT) \
		--quiet --strict-host-key-checking=no \
		--command="sudo systemctl start $(GW_SERVICE)" 2>/dev/null \
		&& echo "  Publisher started" \
		|| echo "  WARNING: Could not start publisher (is VM running?)"
	@echo "  Publisher resumed."

# ------------------------------------------------------------------
# 5) Tear down everything
# ------------------------------------------------------------------
gw-infra-down:
	@echo ""
	@echo "=== GRAPH WAVENET: TEARING DOWN ==="
	@echo ""
	@echo "[1/5] Cancelling Dataflow job..."
	@JOB_ID=$$(gcloud dataflow jobs list \
		--project=$(PROJECT) --region=$(REGION) \
		--filter="name=$(GW_DF_JOB) AND (state=Running OR state=Draining)" \
		--format="value(JOB_ID)" --limit=1 2>/dev/null); \
	if [ -n "$$JOB_ID" ]; then \
		gcloud dataflow jobs cancel "$$JOB_ID" \
			--project=$(PROJECT) --region=$(REGION) --quiet; \
		echo "  Cancelled job $$JOB_ID"; \
	else \
		echo "  No active Dataflow job"; \
	fi
	@echo ""
	@echo "[2/5] Deleting publisher VM..."
	-@gcloud compute instances delete $(GW_VM) --zone=$(ZONE) --project=$(PROJECT) --quiet 2>/dev/null \
		&& echo "  VM deleted" || echo "  VM not found"
	@echo ""
	@echo "[3/5] Deleting Pub/Sub..."
	-@gcloud pubsub subscriptions delete $(GW_SUB) --project=$(PROJECT) --quiet 2>/dev/null \
		&& echo "  Subscription deleted" || echo "  Not found"
	-@gcloud pubsub topics delete $(GW_TOPIC) --project=$(PROJECT) --quiet 2>/dev/null \
		&& echo "  Topic deleted" || echo "  Not found"
	@echo ""
	@echo "[4/5] Cleaning GCS staging..."
	-@gsutil rm -r $(GW_BUCKET) 2>/dev/null && echo "  Publisher staging deleted" || true
	-@gsutil rm -r $(GW_DF_BUCKET) 2>/dev/null && echo "  Dataflow staging deleted" || true
	@echo ""
	@echo "[5/5] BigQuery dataset preserved."
	@echo "  To delete: bq rm -r -f $(PROJECT):$(GW_BQ_DS)"
	@echo ""
	@echo "=== TEARDOWN COMPLETE ==="

# ------------------------------------------------------------------
# Status and logs
# ------------------------------------------------------------------
gw-status:
	@echo ""
	@echo "=== GRAPH WAVENET: STATUS ==="
	@echo ""
	@echo "--- Publisher VM ---"
	@gcloud compute instances describe $(GW_VM) --zone=$(ZONE) --project=$(PROJECT) \
		--format="table(name, status, networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null \
		|| echo "  VM not found"
	@echo ""
	@gcloud compute ssh $(GW_VM) --zone=$(ZONE) --project=$(PROJECT) \
		--quiet --strict-host-key-checking=no \
		--command="sudo systemctl is-active $(GW_SERVICE) 2>/dev/null && echo '  Publisher: RUNNING' || echo '  Publisher: STOPPED'" 2>/dev/null \
		|| echo "  Cannot reach VM"
	@echo ""
	@echo "--- Pub/Sub ---"
	@gcloud pubsub subscriptions describe $(GW_SUB) --project=$(PROJECT) \
		--format="value(messageRetentionDuration)" 2>/dev/null \
		|| echo "  Subscription not found"
	@echo ""
	@echo "--- Dataflow ---"
	@gcloud dataflow jobs list \
		--project=$(PROJECT) --region=$(REGION) \
		--filter="name=$(GW_DF_JOB)" \
		--format="table(JOB_ID, NAME, STATE, START_TIME)" \
		--limit=3 2>/dev/null \
		|| echo "  No Dataflow jobs found"
	@echo ""
	@echo "--- BigQuery ---"
	@bq --project_id=$(PROJECT) show --format=prettyjson $(GW_BQ_DS).$(GW_BQ_TABLE) 2>/dev/null \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print('  Rows: ' + d.get('numRows','0') + '  Size: ' + d.get('numBytes','0') + ' bytes')" 2>/dev/null \
		|| echo "  Table not found"
	@echo ""

gw-logs:
	@echo "Fetching recent logs..."
	@JOB_ID=$$(gcloud dataflow jobs list \
		--project=$(PROJECT) --region=$(REGION) \
		--filter="name=$(GW_DF_JOB) AND state=Running" \
		--format="value(JOB_ID)" --limit=1 2>/dev/null); \
	if [ -n "$$JOB_ID" ]; then \
		echo "--- Dataflow job $$JOB_ID ---"; \
		gcloud logging read "resource.type=dataflow_step AND resource.labels.job_id=$$JOB_ID" \
			--project=$(PROJECT) --limit=50 --format="table(timestamp, textPayload)" \
			--freshness=30m --order=asc; \
	else \
		echo "No running Dataflow job. Showing publisher logs:"; \
		gcloud compute ssh $(GW_VM) --zone=$(ZONE) --project=$(PROJECT) \
			--quiet --strict-host-key-checking=no \
			--command="sudo journalctl -u $(GW_SERVICE) -n 30 --no-pager" 2>/dev/null \
			|| echo "Cannot reach publisher VM"; \
	fi

# ------------------------------------------------------------------
# Help
# ------------------------------------------------------------------
help:
	@echo ""
	@echo "  Graph WaveNet Dataset Pipeline"
	@echo "  =============================="
	@echo ""
	@echo "    make gw-infra-up         Provision VM + Pub/Sub + BQ table"
	@echo "    make gw-start-ingestion  Start poller + launch Dataflow"
	@echo "    make gw-start-dataflow   Launch Dataflow only (poller already running)"
	@echo "    make gw-pause            Pause poller (Dataflow idles)"
	@echo "    make gw-resume           Resume poller"
	@echo "    make gw-status           Status of all components"
	@echo "    make gw-logs             Tail Dataflow or poller logs"
	@echo "    make gw-infra-down       Tear down everything"
	@echo ""
