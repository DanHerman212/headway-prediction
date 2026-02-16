# ============================================================
#  Streaming Integration Test — Make Targets
# ============================================================
#
#  Workflow:
#    make deploy-ingestion     # provision VM + Pub/Sub (one-time)
#    make start-ingestion      # start the poller on the VM
#    make start-integration    # run streaming pipeline locally
#    make check-firestore      # monitor Firestore output
#    make stop-integration     # kill the local pipeline
#    make stop-ingestion       # stop the poller on the VM
#    make teardown             # delete VM + Pub/Sub resources
#
# ============================================================

PROJECT     ?= realtime-headway-prediction
ZONE        ?= us-east1-b
VM_NAME     ?= gtfs-poller
TOPIC       ?= gtfs-rt-ace
SUBSCRIPTION ?= gtfs-rt-ace-sub
SUB_FULL    ?= projects/$(PROJECT)/subscriptions/$(SUBSCRIPTION)

# PID file for the local pipeline process
PID_FILE    := .streaming-pipeline.pid

# ----------------------------------------------------------
#  Infrastructure
# ----------------------------------------------------------

.PHONY: deploy-ingestion
deploy-ingestion: ## Provision Pub/Sub + GCE VM and deploy the poller
	bash infra/deploy_poller.sh

.PHONY: teardown
teardown: stop-integration stop-ingestion ## Tear down VM + Pub/Sub resources
	@echo "--- Deleting VM $(VM_NAME) ---"
	-gcloud compute instances delete $(VM_NAME) \
		--zone=$(ZONE) --project=$(PROJECT) --quiet
	@echo "--- Deleting subscription $(SUBSCRIPTION) ---"
	-gcloud pubsub subscriptions delete $(SUBSCRIPTION) --project=$(PROJECT) --quiet
	@echo "--- Deleting topic $(TOPIC) ---"
	-gcloud pubsub topics delete $(TOPIC) --project=$(PROJECT) --quiet
	@echo "Teardown complete."

# ----------------------------------------------------------
#  Ingestion (poller on VM)
# ----------------------------------------------------------

.PHONY: start-ingestion
start-ingestion: ## Start the poller service on the VM
	bash infra/poller_control.sh start

.PHONY: stop-ingestion
stop-ingestion: ## Stop the poller service on the VM
	bash infra/poller_control.sh stop

.PHONY: ingestion-status
ingestion-status: ## Check poller service status
	bash infra/poller_control.sh status

.PHONY: ingestion-logs
ingestion-logs: ## Tail poller logs (Ctrl-C to stop)
	bash infra/poller_control.sh logs -f

# ----------------------------------------------------------
#  Streaming Pipeline (local DirectRunner)
# ----------------------------------------------------------

.PHONY: start-integration
start-integration: ## Start the streaming pipeline locally (background)
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Pipeline already running (PID $$(cat $(PID_FILE)))"; \
	else \
		echo "Starting streaming pipeline..."; \
		python -m pipelines.beam.streaming.streaming_pipeline \
			--input_subscription $(SUB_FULL) \
			--project_id $(PROJECT) \
			2>&1 | tee streaming-pipeline.log & \
		echo $$! > $(PID_FILE); \
		echo "Pipeline started (PID $$(cat $(PID_FILE))). Logs: streaming-pipeline.log"; \
	fi

.PHONY: stop-integration
stop-integration: ## Stop the local streaming pipeline
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Stopping pipeline (PID $$PID)..."; \
			kill $$PID; \
			sleep 2; \
			kill -0 $$PID 2>/dev/null && kill -9 $$PID || true; \
			echo "Pipeline stopped."; \
		else \
			echo "Pipeline not running (stale PID file)."; \
		fi; \
		rm -f $(PID_FILE); \
	else \
		echo "No pipeline PID file found."; \
	fi

.PHONY: integration-logs
integration-logs: ## Tail the local pipeline log
	tail -f streaming-pipeline.log

# ----------------------------------------------------------
#  Monitoring
# ----------------------------------------------------------

.PHONY: check-firestore
check-firestore: ## One-shot Firestore window snapshot
	python scripts/check_firestore_windows.py --project $(PROJECT)

.PHONY: watch-firestore
watch-firestore: ## Continuously poll Firestore (Ctrl-C to stop)
	python scripts/check_firestore_windows.py --project $(PROJECT) --watch --interval 30

# ----------------------------------------------------------
#  Help
# ----------------------------------------------------------

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
