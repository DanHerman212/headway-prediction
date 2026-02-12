# Finalize Prediction Service

## Overview

This document tracks the end-to-end work required to move from production training to a fully deployed, monitored, and self-retraining headway prediction service. The work is organized into seven sequential phases, each gated on the successful completion of the previous phase.

**Project:** `realtime-headway-prediction`  
**Region:** `us-east1`  
**Model:** Temporal Fusion Transformer (TFT) — quantile regression (P10/P50/P90)  
**Orchestrator:** ZenML → Vertex AI Pipelines  
**Infra:** GCP (Vertex AI, Dataflow, Pub/Sub, Firestore, Artifact Registry, Compute Engine)

---

## Phase 1: Validate Production Training Feedback Loops

**Status:** In Progress — awaiting evaluation step completion  
**Goal:** Confirm training pipeline runs end-to-end with correct experiment tracking, evaluation metrics, and visualization artifacts.

### What's Being Tested

- **Experiment tracking** — `VertexAIMetricsCallback` streams epoch-level `train_loss` / `val_loss` to Vertex AI Experiments via `aiplatform.log_time_series_metrics()`. Final `best_val_loss` logged via `aiplatform.log_metrics()`. **Confirmed working** — metrics visible in Vertex AI console.
- **TensorBoard integration** — `TensorBoardLogger` writes per-step curves locally, then uploads to `gs://mlops-artifacts-realtime-headway-prediction/tensorboard_logs/`. Linked Vertex AI TensorBoard instance (`8313539359009669120`) provides monitoring. **Note:** Vertex AI managed TensorBoard has limited views compared to running locally — TFT-specific plots (interpretations, attention weights) are not rendered. These can be retrieved manually via `model.plot_interpretation()` / `model.plot_prediction()` if needed.
- **Evaluation step** — Computes MAE and sMAPE on P50 quantile forecasts, generates rush hour visualization (A/C/E lines with confidence bands), logs metrics + images to both Vertex AI Experiments and TensorBoard, and uploads all artifacts to GCS. **Awaiting completion.**
- **Vizier HPO loop** — When `--use-vizier-params` is enabled, best hyperparameters from the latest Vizier study are fetched and applied to the training config via `OmegaConf.update()`.

### Validation Criteria

- [x] Pipeline completes training on Vertex AI (A100 GPU, `a2-highgpu-1g`)
- [x] Epoch-level metrics stream to Vertex AI Experiments in real time
- [x] TensorBoard logs upload to GCS
- [x] Vizier param injection applies overrides correctly when enabled
- [ ] Evaluation step completes — `test_mae` and `test_smape` metrics appear in Vertex AI Experiments
- [ ] Rush hour evaluation plots (HTML + PNG) are accessible at `gs://mlops-artifacts-realtime-headway-prediction/tensorboard_logs/evaluation/`
- [ ] Review evaluation outputs to determine if model quality is sufficient for deployment

### Commands

```bash
# Standard training run
python mlops_pipeline/run.py --mode training

# Training with Vizier best params
python mlops_pipeline/run.py --mode training --use-vizier-params

# Monitor in Vertex AI TensorBoard
gcloud ai tensorboards open \
  --tensorboard=8313539359009669120 \
  --region=us-east1 \
  --project=realtime-headway-prediction
```

### Next Step

Once evaluation completes: review rush hour plots and metrics, then proceed to Phase 2.

---

## Phase 2: Export Model Artifacts & Register in Vertex AI

**Status:** Not Started  
**Goal:** Persist model + serving metadata so the prediction endpoint can load everything it needs without any training dependencies.

### Artifacts to Export

| Artifact | Description | Format |
|---|---|---|
| Model weights | TFT `state_dict` | `model.pt` (PyTorch) |
| Dataset parameters | Encoders, normalizer params, categorical mappings | `dataset_params.json` (from `training_dataset.get_parameters()`) |
| Processing config | Feature columns, encoder/prediction lengths, group IDs | `processing_config.yaml` (copy of `conf/processing/default.yaml`) |
| Model config | Architecture hyperparameters | `model_config.yaml` (copy of `conf/model/tft.yaml`) |

### Tasks

- [ ] Add artifact export logic to training pipeline — save model + metadata to `gs://mlops-artifacts-realtime-headway-prediction/models/{run_id}/`
- [ ] Register model in Vertex AI Model Registry with version metadata (run ID, training metrics, Vizier params if used)
- [ ] Verify model can be loaded from GCS artifacts alone (no training pipeline dependencies)

---

## Phase 3: Build & Deploy Stateless Prediction Endpoint

**Status:** Not Started  
**Goal:** Deploy the TFT model to a Vertex AI Prediction Endpoint with a stateless Custom Prediction Routine (CPR).

### Design Decision

The CPR is **fully stateless**. It receives a complete feature payload containing the full encoder window (20 timesteps) from the Dataflow pipeline. No external lookups or state management in the serving layer.

### Architecture

```
Dataflow Request (JSON — full 20-step encoder window + features)
  → Custom Prediction Routine
    → preprocess(): validate schema, reconstruct TimeSeriesDataSet, create DataLoader
    → predict(): model.forward() → quantile tensor (1, 1, 3)
    → postprocess(): map to {p10, p50, p90}, attach group_id + timestamp + model version
  → Response (JSON)
```

### Request / Response Contract

```json
// Request
{
  "group_id": "A_northbound_14st",
  "route_id": "A",
  "observations": [
    {
      "time_idx": 14200,
      "service_headway": 5.2,
      "preceding_train_gap": 4.8,
      "upstream_headway_14th": 5.5,
      "travel_time_14th": 2.1,
      "travel_time_23rd": 1.8,
      "travel_time_34th": 2.3,
      "hour_sin": 0.87,
      "hour_cos": -0.5,
      "regime_id": "AM_RUSH",
      "track_id": "express",
      "preceding_route_id": "A",
      "empirical_median": 5.0,
      "stops_at_23rd": 1.0
      // ... deviation features
    }
    // ... 20 total observations (encoder window)
  ]
}

// Response
{
  "group_id": "A_northbound_14st",
  "prediction": {
    "headway_p10": 4.1,
    "headway_p50": 5.3,
    "headway_p90": 7.2
  },
  "metadata": {
    "model_version": "v1.0-vizier",
    "timestamp": "2026-02-12T14:30:00Z"
  }
}
```

### Tasks

- [ ] Implement `Predictor` class (`load()`, `preprocess()`, `predict()`, `postprocess()`)
- [ ] Build serving container — `Dockerfile.serving` extending Vertex AI prediction base, with `pytorch-forecasting` (no Lightning trainer, no ZenML)
- [ ] Deploy to Vertex AI endpoint (CPU, `n1-standard-4`, 1–5 replicas)
- [ ] Test with synthetic full-window requests — validate response schema, P10 < P50 < P90, values in reasonable headway range

### Serving Config

| Setting | Value |
|---|---|
| Machine type | `n1-standard-4` (CPU inference) |
| Min replicas | 1 |
| Max replicas | 5 |
| Model format | PyTorch `state_dict` + dataset params |
| Container | Custom CPR image in `us-east1-docker.pkg.dev/realtime-headway-prediction/mlops-images/serving:latest` |

---

## Phase 4: Real-Time Streaming Pipeline & Integration Test

**Status:** Not Started  
**Goal:** Build the Dataflow streaming pipeline that processes GTFS-RT events, accumulates encoder context, and calls the prediction endpoint. Validate end-to-end with live MTA data.

### Architecture

```
MTA GTFS-RT Feed (Protobuf)
  → Poller Service (Compute Engine VM)
    → Parse protobuf, extract trip updates for A/C/E lines
    → Publish structured events to Pub/Sub
  → Pub/Sub topic: `gtfs-rt-events`
    → Dataflow Streaming Pipeline (Apache Beam)
      → Stateful DoFn: accumulate events per group_id
      → Per-group window: buffer last 20 observations
      → Feature engineering (same transforms as training: headway calc, hour_sin/cos, deviations, stops_at_23rd)
      → Once window is full → call Prediction Endpoint (REST)
      → Write predictions → Firestore (for app consumption)
      → Write predictions + actuals → BigQuery (for monitoring)
```

### Key Design: Dataflow Owns the State

The Dataflow pipeline maintains the rolling 20-event encoder window per `group_id` using Beam's stateful processing (`@beam.DoFn` with `BagState` or `CombiningValueState`). This means:

- The prediction endpoint stays completely stateless
- The feature engineering logic (headway computation, cyclical time features, deviations) lives in the Beam pipeline, reusing transforms from `data_processing.py`
- Predictions are written to **Firestore** for real-time app consumption and **BigQuery** for monitoring/analytics

### Tasks

- [ ] **Build poller service** — Compute Engine VM running a Python process that polls MTA GTFS-RT feed, parses protobuf, publishes to Pub/Sub
- [ ] **Create Pub/Sub topic** — `gtfs-rt-events`
- [ ] **Build Dataflow streaming pipeline** — Stateful Beam pipeline with per-group accumulation, feature engineering, endpoint calling, dual-write to Firestore + BigQuery
- [ ] **Deploy Dataflow job** — Streaming mode, autoscaling workers
- [ ] **Integration test** — Run end-to-end against live MTA feed for 1 hour during rush hour:
  - Predictions generated for all target routes
  - P50 forecasts are sane (within training-time MAE range)
  - End-to-end latency: event ingestion → prediction in Firestore < 5 seconds
  - No dropped events or Dataflow errors
  - Predictions land in both Firestore and BigQuery

### Validation Criteria

- [ ] Predictions generated for all target routes within the test window
- [ ] P50 prediction error on live data is within acceptable range of offline test metrics
- [ ] No dropped events or timeout errors during the test window
- [ ] Prediction results land in Firestore and BigQuery
- [ ] End-to-end latency < 5 seconds

---

## Phase 5: Model Monitoring Service

**Status:** Not Started  
**Goal:** Detect data drift, prediction quality degradation, and model staleness to trigger retraining.

### Monitoring Signals

| Signal | Method | Threshold |
|---|---|---|
| **Feature drift** | Distribution comparison (PSI / KS-test) on input features vs. training distribution | PSI > 0.2 |
| **Prediction drift** | Monitor P50 prediction distribution shift over rolling windows | KS p-value < 0.05 |
| **Performance degradation** | Compare predicted headways against actuals (from BigQuery prediction logs) | MAE increase > 20% over baseline |
| **Model staleness** | Time since last training run | > 7 days without retraining |

### Architecture

```
BigQuery (prediction logs + actuals from Dataflow)
  → Monitoring Service (Cloud Run, scheduled)
    → Compute drift metrics
    → Compute prediction quality metrics
    → Evaluate alert conditions
  → If alert triggered:
    → Publish to Pub/Sub topic: `model-retraining-triggers`
    → Log alert to Vertex AI Experiments
    → Send notification (email / Slack webhook)
```

### Tasks

- [ ] **Build monitoring service** — Scheduled Cloud Run job that queries prediction logs from BigQuery, computes drift and quality metrics
- [ ] **Define alert rules** — Configurable thresholds for each monitoring signal
- [ ] **Create Pub/Sub topic** — `model-retraining-triggers` for downstream CI/CD integration
- [ ] **Dashboard** — Monitoring metrics visible in Vertex AI or a Looker Studio dashboard
- [ ] **Test with synthetic drift** — Inject modified data to confirm alerts fire correctly

---

## Phase 6: CI/CD for Automated Retraining

**Status:** Not Started  
**Goal:** Close the feedback loop — monitoring alerts automatically trigger a new training cycle via CI/CD.

### Architecture

```
Pub/Sub: `model-retraining-triggers`
  → Cloud Function (trigger handler)
    → Create a Pull Request on GitHub with:
      - Updated training data reference (latest partition)
      - Trigger metadata (drift metrics, alert reason)
  → GitHub Actions CI/CD
    → Lint + unit tests
    → On PR merge:
      → Build training container → push to Artifact Registry
      → Trigger ZenML pipeline on Vertex AI
      → Pipeline: ingest → process → train → evaluate
      → If eval metrics pass acceptance gate:
        → Register new model version (Phase 2 flow)
        → Deploy to endpoint (canary → full traffic)
      → If eval metrics fail:
        → Rollback, notify team
```

### Tasks

- [ ] **Pub/Sub → Cloud Function trigger** — Listens on `model-retraining-triggers`, creates a GitHub PR with retraining context
- [ ] **GitHub Actions workflow** — On PR merge to `main`:
  - Build and push Docker image via Cloud Build (`cloudbuild_hpo.yaml` pattern)
  - Trigger `python mlops_pipeline/run.py --mode training --use-vizier-params`
  - Run evaluation gate: fail the workflow if `test_mae` exceeds threshold
- [ ] **Model promotion logic** — Automated canary deployment: new model serves 10% traffic, promote to 100% if online metrics hold
- [ ] **Rollback mechanism** — If canary metrics degrade, automatically revert to previous model version on the endpoint
- [ ] **Audit trail** — Every retraining cycle logged to Vertex AI Experiments with trigger reason, data snapshot, metrics, and deployment outcome

### CI/CD Config (Planned)

| Component | Tool |
|---|---|
| Container build | Cloud Build (`infra/cloudbuild_hpo.yaml` as template) |
| Pipeline trigger | ZenML CLI or GitHub Actions `workflow_dispatch` |
| Model registry | Vertex AI Model Registry |
| Deployment | Vertex AI Prediction Endpoint (traffic split) |
| Notifications | Pub/Sub → Slack / email |

---

## Dependency Graph

```
Phase 1 (Training Validation)          ← CURRENT: awaiting evaluation step
  └──► Phase 2 (Export Artifacts + Model Registry)
         └──► Phase 3 (Stateless Prediction Endpoint)
                └──► Phase 4 (Dataflow Streaming Pipeline + Integration Test)
                       └──► Phase 5 (Model Monitoring)
                              └──► Phase 6 (CI/CD Automated Retraining)
                                     └──► ♻ Phase 1 (closed loop)
```

Each phase gates on the previous. Phase 6 closes the loop back to Phase 1, creating a fully automated MLOps lifecycle.

---

## Current Status (Feb 12, 2026)

- **Phase 1** — Training complete, experiment tracking confirmed working. Evaluation step running. Pending review of rush hour plots and test metrics before proceeding.
- **Phases 2–6** — Not started. Deployment work begins after evaluation review.
