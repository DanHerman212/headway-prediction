# Deployment Action List — February 19, 2026

**Pre-condition:** Training pipeline completed successfully (exit code 0).

---

## Step 1: Export Model Artifacts
- [x] ONNX export + dataset_params.json + GCS upload — handled by `register_model` step in pipeline (already ran)
- [x] Serving container pushed to Artifact Registry (`headway-serving:latest`)

## Step 2: Deploy to Vertex AI Prediction Endpoint + Model Monitoring
- [ ] Run `python scripts/deploy_endpoint.py --dry-run` to preview actions
- [ ] Run `python scripts/deploy_endpoint.py` to:
  - Find latest `headway-tft` model in Vertex AI Model Registry
  - Create or reuse `headway-prediction-endpoint`
  - Deploy model (n1-standard-4, CPU, 1 replica)
  - Generate monitoring baseline CSV from training parquet → GCS
  - Create Model Monitoring job (feature drift + skew, 1h interval)
  - Smoke test with synthetic prediction request
- [ ] Verify endpoint in console: https://console.cloud.google.com/vertex-ai/endpoints?project=realtime-headway-prediction

## Step 3: Deploy Production Ingestion Infrastructure
- [ ] `make deploy-infra` — provisions:
  - Pub/Sub topic (`gtfs-rt-ace`) + subscription (`gtfs-rt-ace-sub`, 7-day retention)
  - Poller VM (`gtfs-poller`, e2-small, Debian 12, systemd service)
  - GCS staging buckets for poller code + Dataflow
- [ ] `make start-ingestion` — starts:
  - Waits for poller VM startup script to complete
  - Verifies messages flowing on Pub/Sub
  - Launches Dataflow streaming pipeline (`DataflowRunner`, n1-standard-2, max 3 workers)
  - Pipeline: Pub/Sub → arrival detection → feature engineering → window buffer (20-obs warmup) → prediction endpoint → Firestore
- [ ] Monitor: `make prod-status` / `make prod-logs`
- [ ] Confirm first predictions in Firestore after ~10-15 min warmup

## Step 4: Refactor Firestore Sink (Predictions Only, Append-All)
- [ ] Create new collection `predictions` — each prediction is a new document (no overwrites)
- [ ] Fields: `group_id`, `headway_p10`, `headway_p50`, `headway_p90`, `predicted_at`, `model_version`, `last_observation_time_idx`
- [ ] Remove raw observation window from the write (predictions only)
- [ ] Decide: keep or remove existing `headway_windows` collection writes

## Step 5: Live Evaluation (MAE / sMAPE)
- [ ] Implement `EvaluatePredictionFn` DoFn — stateful, per-group:
  - On each new arrival, retrieve previous prediction from Beam state
  - New arrival's `service_headway` = ground truth for previous prediction
  - Compute MAE = |actual − predicted_p50|, sMAPE = 2|actual − pred| / (|actual| + |pred|)
- [ ] Write evaluation records to Firestore `evaluation_metrics` collection
- [ ] Verify metrics arriving after warm-up + first prediction cycle

## Operations Reference

| Command | Action |
|---|---|
| `make deploy-infra` | Provision VM + Pub/Sub + GCS |
| `make start-ingestion` | Start poller + Dataflow |
| `make pause-ingestion` | Stop poller + drain Dataflow (Pub/Sub retains 7 days) |
| `make restart-ingestion` | Restart poller + relaunch Dataflow (processes backlog) |
| `make teardown` | Delete all infrastructure (except prediction endpoint) |
| `make prod-status` | Check VM / Pub/Sub / Dataflow status |
| `make prod-logs` | Tail Dataflow worker logs or poller journal |

---

## Infrastructure Reference

| Resource | Value |
|---|---|
| Project | `realtime-headway-prediction` |
| Region | `us-east1` |
| Serving Image | `us-east1-docker.pkg.dev/realtime-headway-prediction/mlops-images/headway-serving:latest` |
| Artifact URI | `gs://mlops-artifacts-realtime-headway-prediction/serving/v1/` |
| Endpoint Name | `headway-prediction-endpoint` |
| Pub/Sub Topic | `projects/realtime-headway-prediction/topics/gtfs-rt-ace` |
| Firestore DB | `headway-streaming` |
| Firestore Collections | `predictions`, `evaluation_metrics` |
| Service Account | `mlops-sa@realtime-headway-prediction.iam.gserviceaccount.com` |
