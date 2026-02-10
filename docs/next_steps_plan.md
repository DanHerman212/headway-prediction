# Headway Prediction Pipeline — Next Steps Plan

**Date:** February 10, 2026  
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
| Experiment tracking | Transitioning from MLflow → TensorBoard |

---

## Phase 1 — Evaluation & Experiment Tracking

### 1a. Switch to TensorBoard

**Goal:** Replace MLflow as the primary experiment tracker during training. TensorBoard provides richer training diagnostics with minimal engineering overhead.

**Why TensorBoard:**
- First-class Lightning integration — `TensorBoardLogger` works out of the box
- Scalars, gradient histograms, computational graph, and profiler built-in
- Native Vertex AI TensorBoard — managed, shareable, persistent
- No custom proxy classes needed (eliminates `SafeMLFlowLogger` / `_ExperimentProxy` workaround)

**What to log:**

| Category | Signals | Implementation |
|---|---|---|
| **Scalars** | train_loss, val_loss, val_MAE, val_sMAPE, learning_rate | Automatic via Lightning callbacks |
| **Gradient Histograms** | Weight & gradient distributions per layer | `Trainer(log_every_n_steps=50)` + override `training_step` to call `self.logger.experiment.add_histogram()` for named parameters |
| **Model Profiler** | GPU utilization, kernel execution times, data loading bottlenecks, memory allocation | `Trainer(profiler="pytorch")` with `torch.profiler.tensorboard_trace_handler` to emit Chrome trace format viewable in TensorBoard's Profile tab |
| **Hyperparameters** | All model/training config | `self.logger.log_hyperparams()` — appears in TensorBoard HParams tab |

**Tasks:**
- [ ] Replace `SafeMLFlowLogger` in `train_model.py` with Lightning `TensorBoardLogger` writing to GCS (`gs://...tensorboard_logs/`)
- [ ] Enable Vertex AI TensorBoard instance for persistent, shareable dashboards
- [ ] Configure `Trainer(profiler="pytorch")` with `schedule(wait=1, warmup=1, active=3, repeat=2)` to capture profiler traces without overhead on every step
- [ ] Verify gradient histograms populate — may require `log_graph=True` and explicit `add_histogram` calls in `on_after_backward` hook
- [ ] Keep direct `mlflow.log_metric()` calls in `evaluate_model.py` for final test metrics (MLflow continues as model registry, not training tracker)

### 1b. Improve Prediction Visualization

**Goal:** Overhaul the rush hour prediction plot for interpretability and presentation quality.

**Current issues:**
- Subplots not labeled with specific `train_id`
- X-axis shows raw time index, not human-readable hours of service
- Default matplotlib color scheme, minimal formatting

**Improvements:**

| Aspect | Change |
|---|---|
| **Subplot titles** | Include `train_id` (e.g., "Train A-12 · Northbound · Grand Central → 125 St") |
| **X-axis** | Convert time index to hours of service (e.g., `06:00`, `06:15`, `06:30`…) using `mdates.DateFormatter` or manual tick labels derived from `time_idx` → datetime mapping |
| **Color scheme** | Muted, accessible palette — prediction band in translucent blue, actuals as dark solid line, quantile bounds as dashed. Consider colorblind-safe palette (e.g., Okabe-Ito or seaborn `colorblind`) |
| **Layout** | Larger figure size, tighter subplot spacing, legend outside plot area, grid lines at major hours |
| **Annotations** | Mark peak/off-peak regions with light background shading |

**Output format — explore interactive HTML:**
- **Option A:** Static PNG stored in GCS (current approach, simple, works everywhere)
- **Option B:** Plotly/Bokeh HTML — interactive zoom/hover with train_id tooltips, renderable from TensorBoard (via `SummaryWriter.add_custom_scalars` or iframe embed) or served directly from GCS as a signed URL
- **Recommendation:** Start with Plotly for the evaluation plot. Store `.html` alongside `.png` in GCS. TensorBoard can render HTML via the Text plugin or a custom iframe. Vertex AI Pipelines UI can display HTML artifacts natively.

**Tasks:**
- [ ] Refactor `RushHourVisualizer` (or equivalent in `evaluate_model.py`) to accept `train_id` metadata and map `time_idx` to datetime
- [ ] Switch to Plotly for interactive output with hover tooltips (train_id, predicted vs actual, quantile range)
- [ ] Apply accessible color palette and improved layout
- [ ] Store both `.png` (backward compat) and `.html` to GCS artifact path
- [ ] Verify HTML renders in Vertex AI Pipelines UI and TensorBoard

---

## Phase 2 — Hyperparameter Optimization (Vertex AI Vizier)

**Goal:** Establish the golden path — Vizier HPO as a pipeline step, followed by a full training run on the best params.

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

**Golden Path (pipeline flow):**

```
vizier_hpo_step          # Create Vizier study, run N trials (reduced epochs, early stopping)
       │
       ▼
train_best_params_step   # Full training run with best hyperparameters from Vizier
       │
       ▼
evaluate_model_step      # Test metrics + prediction plots (TensorBoard + GCS)
       │
       ▼
register_model_step      # Save best model to Vertex AI Model Registry (or MLflow Registry)
       │
       ▼
deploy_model_step        # Deploy to Vertex AI Prediction Endpoint + attach Model Monitor
```

**Trial configuration:**
- Each trial: ~15 epochs, early stopping patience 5
- Objective metric: `val_MAE` (interpretable, aligns with business metric)
- ~20-30 trials, 5 parallel at a time → ~25-30 min wall time
- Best params fed into full training run (100+ epochs with early stopping patience 15)

**Tasks:**
- [ ] Implement `vizier_hpo_step` as a ZenML step that creates a Vizier `Study`, submits `Trial`s, and returns best parameters
- [ ] Wire Vizier output into training step config via Hydra overrides
- [ ] Implement `register_model_step` — save model checkpoint + dataset params (encoders, scalers) to Vertex AI Model Registry
- [ ] Implement `deploy_model_step` — create/update Vertex AI Endpoint with traffic splitting (canary → full)
- [ ] Attach Vertex AI Model Monitoring to the endpoint (feature drift + prediction drift)

---

## Phase 3 — Model Deployment & Monitoring

**Goal:** Serve real-time headway predictions via a Vertex AI Prediction Endpoint with continuous monitoring.

### Serving

- **Serving container:** FastAPI app that loads the TFT model + dataset parameters (encoders, scalers)
- **Input:** JSON request with route_id, direction, recent observations
- **Preprocessing:** Apply same `GroupNormalizer`, `NaNLabelEncoder`, feature transformations as training
- **Output:** Quantile predictions (P10, P50, P90) in real headway minutes
- **Infrastructure:** Vertex AI Endpoint on `n1-standard-4` + 1x T4 GPU (cost-effective for inference)
- **Traffic splitting:** Vertex AI supports gradual rollout (10% → 100%)

### Monitoring (three layers)

| Layer | What | Implementation | Alert Threshold |
|---|---|---|---|
| **A — Input Drift** | Feature distribution shift in prediction requests | Log requests to BigQuery, Vertex AI Model Monitoring (feature skew/drift) | PSI > 0.2 on any feature |
| **B — Prediction Quality** | Predicted P50 vs. actual headway (ground truth arrives when next train does) | Cloud Function on new actuals in BigQuery, rolling MAE/sMAPE over 1h/6h/24h | Rolling 6h MAE > 1.5× test MAE |
| **C — Staleness & Retrain** | Time since last training, performance degradation, new route data | Cloud Scheduler → Cloud Function → ZenML pipeline | >30 days or metric regression |

**Dashboard:** Looker Studio or Grafana connected to BigQuery (request volume, latency, drift scores, rolling accuracy)

---

## Phase 4 — Real-Time Event Feed (Future Iteration)

**Owner:** Dan  
**Goal:** Poll GTFS-RT data feed, structure events via Apache Beam transforms, and call the deployed prediction endpoint.

**Architecture:**
- Apache Beam pipeline (Dataflow runner) consuming GTFS-RT feed
- Beam transforms structure raw events into the model's expected input schema (matching training preprocessing)
- Transformed records call the Vertex AI Prediction Endpoint
- Predictions written to BigQuery for downstream consumers

**Integration contract (to align on when we get there):**
- Input schema: which fields the endpoint expects
- Feature transformations: must match training preprocessing (group encoding, normalizer)
- Response format: P10/P50/P90 quantile predictions in minutes

**Dependencies:**
- Prediction endpoint must be deployed (Phase 3) before end-to-end testing
- Encoder/normalizer artifacts must be bundled with the serving container

> **Note:** This is scoped for future iteration. Enough on the plate with Phases 1-3 — we'll refine the event feed design once the golden path is running.

---

## Implementation Order

1. **Phase 1a — TensorBoard integration** — swap experiment tracker, enable scalars + histograms + profiler
2. **Phase 1b — Prediction plot overhaul** — train_id labels, hours of service, Plotly interactive HTML
3. **Phase 2 — Vizier HPO → full training → register → deploy** — establish the golden path end-to-end
4. **Phase 3 — Monitoring** — attach model monitor, set up drift/quality/staleness alerts
5. **Phase 4 — Real-time event feed** — Apache Beam + Dataflow, iterate once the prediction endpoint is live

---

## Infrastructure Reference

| Resource | Value |
|---|---|
| GCP Project | `realtime-headway-prediction` |
| Region | `us-east1` |
| Service Account | `mlops-sa@realtime-headway-prediction.iam.gserviceaccount.com` |
| ZenML Server | `https://zenml-server-gxvzscak4q-ue.a.run.app` |
| MLflow Server | `https://mlflow-server-gxvzscak4q-ue.a.run.app` (retained for model registry + eval metrics) |
| Vertex AI TensorBoard | TBD — create instance for persistent training dashboards |
| MLflow Experiment | `headway_training_pipeline` |
| Artifact Bucket | `gs://mlops-artifacts-realtime-headway-prediction` |
| Docker Base Image | `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` |
| GPU | A100 (`a2-highgpu-1g`) for training, T4 for inference |
| ZenML Stack | `gcp_vertex_stack` |
