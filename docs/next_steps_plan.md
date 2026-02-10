# Headway Prediction Pipeline — Next Steps Plan

**Date:** February 9, 2026  
**Status:** Training pipeline running successfully on Vertex AI (A100 GPU)

---

## Current State

| Component | Status |
|---|---|
| GPU (A100) detection | ✅ Working — `bf16-mixed` precision active |
| Training metrics → MLflow | ✅ train_loss, val_loss, val_MAE, val_sMAPE, lr-Ranger |
| Hyperparameters → MLflow | ✅ Logged at step start |
| Evaluation metrics → MLflow | ✅ test_mae, test_smape, rush_hour_performance.png |
| Docker base image | `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` |
| setuptools pin | `>=70,<78` (preserves `pkg_resources`) |
| Dataset | 6 months of GTFS-RT headway data |
| Pipeline orchestration | ZenML on Vertex AI Pipelines |
| Experiment tracking | MLflow on Cloud Run (MySQL backend, GCS artifact store) |

---

## 1. Expand Dataset (6 → 12 Months)

**Goal:** Double the training data for better generalization across seasonal patterns.

**Tasks:**
- Update training data parquet on GCS with 12 months of data
- Update processing config cutoff dates (`train_end_date`, `val_end_date`, `test_end_date`)
- Evaluate increasing `batch_size` (128 → 256 or 512) — A100 has 80GB VRAM headroom
- Confirm `max_encoder_length` / `max_prediction_length` are appropriate for the longer time horizon
- Monitor memory usage — double data means more batches per epoch

---

## 2. Hyperparameter Optimization (Vertex AI Vizier)

**Goal:** Systematically find optimal TFT hyperparameters using managed parallel trials.

**Why Vizier over Optuna:**
- Native to Vertex AI — no additional infrastructure
- Managed parallelism — run N trials across N separate A100 jobs simultaneously
- State persisted server-side — crash-resistant, resume anytime
- Google's Bayesian optimization (TPE-equivalent)
- Cost-efficient — early stopping of underperforming trials at infrastructure level

**Search Space:**

| Parameter | Current | Search Range | Type |
|---|---|---|---|
| `learning_rate` | 0.001 | 1e-4 → 1e-2 | Log-uniform |
| `hidden_size` | 128 | [64, 128, 256] | Categorical |
| `dropout` | 0.1 | 0.05 → 0.3 | Uniform |
| `attention_head_size` | 4 | [1, 2, 4, 8] | Categorical |
| `hidden_continuous_size` | 64 | [32, 64, 128] | Categorical |
| `batch_size` | 128 | [128, 256, 512] | Categorical |

**Architecture:**
- New ZenML pipeline or step that creates a Vizier study
- Each trial: reduced epoch budget (~15 epochs, early stopping patience 5)
- Objective metric: `val_loss` (or `val_MAE` for interpretability)
- ~20-30 trials, 5 parallel at a time → ~25-30 min wall time
- Best params fed into a full training run

---

## 3. Model Deployment

**Goal:** Serve real-time headway predictions via a Vertex AI Prediction Endpoint.

**Architecture:**
- **Serving container:** FastAPI app that loads the TFT model + dataset parameters (encoders, scalers)
- **Input:** JSON request with route_id, direction, recent observations
- **Preprocessing:** Apply same `GroupNormalizer`, `NaNLabelEncoder`, feature transformations as training
- **Output:** Quantile predictions (P10, P50, P90) in real headway minutes
- **Infrastructure:** Vertex AI Endpoint on `n1-standard-4` + 1x T4 GPU (cost-effective for inference)
- **Traffic splitting:** Vertex AI supports gradual rollout (10% → 100%)

**MLflow Model Registry Integration:**
- Register trained model with version tag after training
- Promote to "Production" stage after evaluation metrics pass thresholds
- Deployment pipeline triggers on stage transition

---

## 4. Model Monitoring

**Three monitoring layers:**

### Layer A — Input Drift (Data Quality)
- Monitor incoming prediction requests for distribution shift
- Key signals: headway value ranges, missing routes, new route IDs, time-of-day distribution changes
- Implementation: Log requests to BigQuery, Vertex AI Model Monitoring (feature skew/drift)
- Alert: PSI > 0.2 on any feature

### Layer B — Prediction Quality (Ground Truth Comparison)
- Compare predicted P50 vs. actual headway (available shortly after — next train arrives)
- Compute rolling MAE/sMAPE over 1h/6h/24h windows
- Implementation: Cloud Function triggered by new actuals in BigQuery
- Alert: Rolling 6h MAE exceeds 1.5× the test MAE from training

### Layer C — Model Staleness & Retraining Triggers
- Track time since last training run
- Trigger retraining when: (a) performance degrades past threshold, (b) >30 days since last training, or (c) new route data available
- Implementation: Cloud Scheduler → Cloud Function → ZenML pipeline

**Dashboard:** Looker Studio or Grafana connected to BigQuery for operational metrics (request volume, latency, drift scores, rolling accuracy)

---

## 5. Real-Time Event Feed (Parallel Workstream)

**Owner:** Dan  
**Goal:** Poll GTFS-RT data feed, structure events, and call the deployed prediction endpoint.

**Integration contract (to align on):**
- Input schema: which fields the endpoint expects
- Feature transformations: must match training preprocessing (group encoding, normalizer)
- Response format: P10/P50/P90 quantile predictions in minutes

**Dependencies:**
- Prediction endpoint must be deployed (Step 3) before end-to-end testing
- Encoder/normalizer artifacts must be bundled with the serving container

---

## Suggested Implementation Order

1. **Dataset expansion** — quickest win, improves model quality
2. **Vizier HPO** — builds on expanded dataset, finds optimal architecture
3. **Deployment** — needed before monitoring or real-time feed can work
4. **Monitoring** — production readiness
5. **Real-time feed integration** — end-to-end system

---

## Infrastructure Reference

| Resource | Value |
|---|---|
| GCP Project | `realtime-headway-prediction` |
| Region | `us-east1` |
| Service Account | `mlops-sa@realtime-headway-prediction.iam.gserviceaccount.com` |
| ZenML Server | `https://zenml-server-gxvzscak4q-ue.a.run.app` |
| MLflow Server | `https://mlflow-server-gxvzscak4q-ue.a.run.app` |
| MLflow Experiment | `headway_training_pipeline` |
| Artifact Bucket | `gs://mlops-artifacts-realtime-headway-prediction` |
| Docker Base Image | `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` |
| GPU | A100 (`a2-highgpu-1g`) for training, T4 for inference |
| ZenML Stack | `gcp_vertex_stack` |
