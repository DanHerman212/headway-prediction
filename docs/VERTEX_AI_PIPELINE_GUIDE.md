# Vertex AI Training Pipeline Guide

This document provides a step-by-step guide to the Vertex AI training pipeline infrastructure, explaining both existing modules that were modified and new modules that were created.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Existing Modules (Modified)](#existing-modules-modified)
5. [New Modules (Created)](#new-modules-created)
6. [Infrastructure Files](#infrastructure-files)
7. [How It All Fits Together](#how-it-all-fits-together)
8. [Running the Pipeline](#running-the-pipeline)
9. [Monitoring & Results](#monitoring--results)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Local Development                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │ run_training │───▶│ Cloud Build  │───▶│ Artifact Registry            │  │
│  │    .sh       │    │ (Dockerfile) │    │ (Container Image)            │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────────┘  │
│         │                                              │                     │
│         │ submit                                       │                     │
│         ▼                                              ▼                     │
│  ┌──────────────┐                          ┌──────────────────────────────┐ │
│  │ kfp_pipeline │─────────────────────────▶│ Vertex AI Pipelines          │ │
│  │    .py       │                          │ (Orchestration)              │ │
│  └──────────────┘                          └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              Google Cloud                                    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Vertex AI Pipelines                               │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │  │ Experiment 1│ │ Experiment 2│ │ Experiment 3│ │ Experiment 4│    │   │
│  │  │ (baseline)  │ │ (dropout)   │ │ (dropout+L2)│ │ (full_reg)  │    │   │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘    │   │
│  │         │               │               │               │            │   │
│  │         └───────────────┴───────────────┴───────────────┘            │   │
│  │                                 │                                     │   │
│  │                                 ▼                                     │   │
│  │                    ┌────────────────────────┐                        │   │
│  │                    │ A100 GPU Training Jobs │                        │   │
│  │                    │ (CustomContainerSpec)  │                        │   │
│  │                    └────────────────────────┘                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ GCS Bucket      │  │ TensorBoard     │  │ Vertex AI Experiments       │  │
│  │ - Training data │  │ - Live metrics  │  │ - Hyperparameters           │  │
│  │ - Checkpoints   │  │ - Loss curves   │  │ - Final metrics             │  │
│  │ - Results       │  │                 │  │ - Run comparison            │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. Google Cloud Setup

```bash
# Install gcloud CLI (if not already installed)
brew install google-cloud-sdk

# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project time-series-478616
```

### 2. Enable Required APIs

```bash
gcloud services enable \
    aiplatform.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com
```

### 3. Python Dependencies

```bash
pip install kfp>=2.0.0 google-cloud-aiplatform>=1.38.0
```

### 4. Create GCS Bucket for Data

```bash
gsutil mb -l us-east1 gs://st-convnet-training-configuration
```

### 5. Upload Training Data

```bash
gsutil -m cp data/*.npy gs://st-convnet-training-configuration/headway-prediction/data/
gsutil -m cp data/*.csv gs://st-convnet-training-configuration/headway-prediction/data/
```

### 6. Create TensorBoard Instance (Optional but Recommended)

```bash
gcloud ai tensorboards create \
    --display-name="Headway Prediction TensorBoard" \
    --project=time-series-478616 \
    --region=us-east1
```

Note the TensorBoard ID returned (e.g., `3732815588020453376`).

---

## Project Structure

```
headway-prediction/
├── src/
│   ├── config.py              # Base model configuration (EXISTING)
│   ├── metrics.py             # Custom Keras metrics (EXISTING)
│   ├── data/
│   │   └── dataset.py         # SubwayDataGenerator (EXISTING)
│   ├── models/
│   │   └── st_convnet.py      # HeadwayConvLSTM model (MODIFIED)
│   └── experiments/           # NEW DIRECTORY
│       ├── __init__.py
│       ├── experiment_config.py   # Experiment definitions (NEW)
│       ├── run_experiment.py      # Training entry point (NEW)
│       └── kfp_pipeline.py        # Kubeflow Pipeline DAG (NEW)
├── Dockerfile                 # Container build spec (NEW)
├── cloudbuild.yaml            # Cloud Build config (NEW)
└── run_training.sh            # CLI wrapper script (NEW)
```

---

## Existing Modules (Modified)

### `src/models/st_convnet.py`

**What it was:** ConvLSTM encoder-decoder model for headway prediction.

**What changed:** Added `spatial_dropout_rate` parameter to enable CuDNN-compatible regularization.

```python
class HeadwayConvLSTM:
    def __init__(
        self,
        config: Config = None,
        spatial_dropout_rate: float = 0.0,  # NEW PARAMETER
    ):
        self.spatial_dropout_rate = spatial_dropout_rate
```

**Why:** Standard Dropout breaks CuDNN optimization for ConvLSTM layers (forces CPU fallback). SpatialDropout3D drops entire feature maps, maintaining CuDNN compatibility while providing regularization.

**Where it's used:** After each ConvLSTM + BatchNormalization block:

```python
x = layers.ConvLSTM2D(...)(x)
x = layers.BatchNormalization()(x)
if self.spatial_dropout_rate > 0:
    x = layers.SpatialDropout3D(self.spatial_dropout_rate)(x)
```

### `src/data/dataset.py`

**No changes required.** The `SubwayDataGenerator` class already supports:
- Loading from local paths
- Creating TensorFlow datasets with batching/prefetching
- Train/validation splitting via index ranges

### `src/config.py`

**No changes required.** The base `Config` class provides model hyperparameters that are extended by `ExperimentConfig`.

### `src/metrics.py`

**No changes required.** Custom metrics (`rmse_seconds`, `r_squared`) are imported and used in training.

---

## New Modules (Created)

### `src/experiments/experiment_config.py`

**Purpose:** Defines the 4 regularization experiments as structured configurations.

**Key Class:**
```python
@dataclass
class ExperimentConfig:
    exp_id: int
    exp_name: str
    description: str
    
    # Regularization parameters
    spatial_dropout_rate: float = 0.0
    weight_decay: float = 0.0
    learning_rate: float = 1e-3
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    
    # Model architecture
    lookback_mins: int = 30
    forecast_mins: int = 15
    filters: int = 32
    num_stations: int = 31
```

**Experiments Defined:**

| ID | Name | Dropout | Weight Decay | Learning Rate |
|----|------|---------|--------------|---------------|
| 1 | baseline | 0.0 | 0.0 | 1e-3 |
| 2 | dropout_only | 0.2 | 0.0 | 1e-3 |
| 3 | dropout_l2 | 0.2 | 1e-4 | 1e-3 |
| 4 | full_regularization | 0.2 | 1e-4 | 3e-4 |

**Helper Functions:**
- `get_experiment(exp_id)` - Retrieve config by ID
- `list_experiments()` - List all available experiments

---

### `src/experiments/run_experiment.py`

**Purpose:** Main training entry point that runs inside the container on Vertex AI.

**Key Components:**

#### 1. VertexExperimentCallback
```python
class VertexExperimentCallback(tf.keras.callbacks.Callback):
    """Logs metrics to Vertex AI Experiments in real-time."""
    
    def on_epoch_end(self, epoch, logs=None):
        aiplatform.log_time_series_metrics(metrics_to_log, step=epoch + 1)
```

Enables live metric streaming to Vertex AI Experiments console.

#### 2. GCS Data Handling
```python
def download_gcs_data(gcs_data_dir: str, local_dir: str) -> str:
    """Download data from GCS to local /tmp for training."""
```

**Why needed:** NumPy can't read directly from `gs://` paths. Data must be downloaded to local disk first.

#### 3. GCS Upload for Artifacts
```python
def upload_to_gcs(local_dir: str, gcs_dir: str):
    """Upload local artifacts (checkpoints, results) to GCS."""
```

**Why needed:** Keras `.keras` format can't save directly to GCS. We save locally then upload.

#### 4. Main Training Flow
```python
def run_experiment(exp_id, data_dir, output_dir, project, location, tensorboard_id):
    # 1. Initialize Vertex AI with experiment tracking
    aiplatform.init(
        project=project,
        location=location,
        experiment=experiment_name,
        experiment_tensorboard=tensorboard_resource
    )
    
    # 2. Start experiment run
    aiplatform.start_run(run_name)
    
    # 3. Log hyperparameters
    aiplatform.log_params({...})
    
    # 4. Build model with regularization
    model = HeadwayConvLSTM(
        config=base_config,
        spatial_dropout_rate=exp_config.spatial_dropout_rate
    ).build_model()
    
    # 5. Train with callbacks
    history = model.fit(train_ds, validation_data=val_ds, callbacks=callbacks)
    
    # 6. Log final metrics
    aiplatform.log_metrics({
        "best_val_rmse_seconds": best_val_rmse,
        ...
    })
    
    # 7. End run
    aiplatform.end_run()
```

---

### `src/experiments/kfp_pipeline.py`

**Purpose:** Defines the Kubeflow Pipeline DAG that orchestrates parallel training jobs.

**Key Components:**

#### 1. Training Component
```python
@dsl.container_component
def train_experiment(
    exp_id: int,
    container_image: str,
    data_dir: str,
    output_dir: str,
    ...
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=container_image,
        command=["python", "-m", "src.experiments.run_experiment"],
        args=[
            "--exp_id", dsl.PIPELINE_TASK_NAME,
            "--data_dir", data_dir,
            ...
        ]
    )
```

Each experiment runs as an independent container with A100 GPU.

#### 2. Pipeline Definition
```python
@dsl.pipeline(name="headway-regularization-sweep")
def regularization_sweep_pipeline(...):
    # Run all 4 experiments in parallel
    exp1 = train_experiment(exp_id=1, ...).set_display_name("exp1_baseline")
    exp2 = train_experiment(exp_id=2, ...).set_display_name("exp2_dropout")
    exp3 = train_experiment(exp_id=3, ...).set_display_name("exp3_dropout_l2")
    exp4 = train_experiment(exp_id=4, ...).set_display_name("exp4_full_reg")
```

#### 3. Pipeline Submission
```python
def submit_pipeline(...):
    job = aiplatform.PipelineJob(
        display_name="headway-regularization-sweep",
        template_path=pipeline_file,
        pipeline_root=pipeline_root,
        ...
    )
    job.submit(service_account=service_account)
```

**Important:** The `service_account` parameter is required for TensorBoard integration.

---

## Infrastructure Files

### `Dockerfile`

```dockerfile
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Entry point is specified in the pipeline, not here
```

**Key decisions:**
- Uses official Vertex AI TensorFlow GPU image (has CUDA, cuDNN pre-installed)
- TensorFlow 2.14 with Python 3.10
- Copies only `src/` directory (not data, notebooks, etc.)

### `cloudbuild.yaml`

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t', '${_IMAGE}:${_TAG}'
      - '-t', '${_IMAGE}:latest'
      - '.'

images:
  - '${_IMAGE}:${_TAG}'
  - '${_IMAGE}:latest'
```

**Key decisions:**
- Builds two tags: timestamped (for versioning) and `latest` (for convenience)
- Uses Artifact Registry (not Container Registry which is deprecated)

### `run_training.sh`

CLI wrapper that simplifies common operations:

```bash
./run_training.sh build-local   # Docker build locally (for testing)
./run_training.sh build-cloud   # Build via Cloud Build
./run_training.sh submit        # Submit pipeline to Vertex AI
./run_training.sh quick         # build-cloud + submit
./run_training.sh all           # Full pipeline with data upload
```

---

## How It All Fits Together

### Data Flow

```
1. Local data files (data/*.npy, data/*.csv)
        │
        ▼ gsutil cp
2. GCS bucket (gs://st-convnet-training-configuration/headway-prediction/data/)
        │
        ▼ download_gcs_data()
3. Container local disk (/tmp/data/)
        │
        ▼ SubwayDataGenerator
4. TensorFlow Dataset (batched, prefetched)
        │
        ▼ model.fit()
5. Trained model + checkpoints (local /tmp/outputs/)
        │
        ▼ upload_to_gcs()
6. GCS outputs (gs://st-convnet-training-configuration/headway-prediction/outputs/)
```

### Execution Flow

```
1. run_training.sh submit
        │
        ▼
2. kfp_pipeline.py compiles DAG to JSON
        │
        ▼
3. aiplatform.PipelineJob.submit() sends to Vertex AI
        │
        ▼
4. Vertex AI creates 4 CustomJob instances (parallel)
        │
        ▼
5. Each job pulls container from Artifact Registry
        │
        ▼
6. Each container runs: python -m src.experiments.run_experiment --exp_id N
        │
        ▼
7. run_experiment.py:
   - Downloads data from GCS
   - Initializes Vertex AI Experiments
   - Builds model with experiment-specific config
   - Trains with callbacks (TensorBoard, Experiments API)
   - Uploads results to GCS
```

### Metric Flow

```
Training Loop
    │
    ├──▶ TensorBoard Callback ──▶ TensorBoard (loss curves)
    │
    ├──▶ VertexExperimentCallback ──▶ Vertex AI Experiments (time-series)
    │
    └──▶ End of training ──▶ aiplatform.log_metrics() (final summary)
```

---

## Running the Pipeline

### Quick Start

```bash
# 1. Build and submit (assumes data already in GCS)
./run_training.sh quick
```

### Full Workflow

```bash
# 1. Build container locally to test
./run_training.sh build-local

# 2. Build container in Cloud Build
./run_training.sh build-cloud

# 3. Submit pipeline
./run_training.sh submit
```

### With Fresh Data Upload

```bash
./run_training.sh all
```

---

## Monitoring & Results

### Vertex AI Pipelines Console
**URL:** https://console.cloud.google.com/vertex-ai/pipelines?project=time-series-478616

Shows:
- Pipeline run status
- Individual experiment status
- DAG visualization
- Logs for each experiment

### TensorBoard
**URL:** https://console.cloud.google.com/vertex-ai/experiments/tensorboard-instances/regions/us-east1/3732815588020453376?project=time-series-478616

Shows:
- Live training curves
- Loss and metric plots
- Comparison across runs

### Vertex AI Experiments
**URL:** https://console.cloud.google.com/vertex-ai/experiments/headway-regularization?project=time-series-478616

Shows:
- All runs with hyperparameters
- Final metrics comparison
- Sortable/filterable table
- Time-series metrics

### GCS Outputs
```bash
# List outputs
gsutil ls gs://st-convnet-training-configuration/headway-prediction/outputs/

# Download results
gsutil -m cp -r gs://st-convnet-training-configuration/headway-prediction/outputs/TIMESTAMP/ ./results/
```

Each experiment outputs:
- `results.json` - Final metrics and config
- `best_model.keras` - Best model checkpoint
- `history.csv` - Training history

---

## Troubleshooting

### Common Issues

#### 1. "NumPy can't read gs:// paths"
**Symptom:** `FileNotFoundError` or `ValueError` when loading data
**Solution:** The `download_gcs_data()` function handles this by copying to `/tmp/data`

#### 2. "Keras can't save to GCS"  
**Symptom:** `ValueError: options is not supported`
**Solution:** Save locally to `/tmp`, then use `upload_to_gcs()` to copy to GCS

#### 3. "TensorBoard integration fails"
**Symptom:** No TensorBoard data appears
**Solution:** Ensure `service_account` is passed to `job.submit()` in kfp_pipeline.py

#### 4. "Experiment.start_run() doesn't exist"
**Symptom:** `AttributeError: 'Experiment' object has no attribute 'start_run'`
**Solution:** Use `aiplatform.start_run()` static method after setting `experiment=` in `aiplatform.init()`

#### 5. "Container build fails"
**Symptom:** Cloud Build errors
**Solution:** 
```bash
# Check build logs
gcloud builds list --limit=5

# Get detailed logs
gcloud builds log BUILD_ID
```

### Viewing Job Logs

```bash
# Get recent pipeline jobs
gcloud ai custom-jobs list --region=us-east1 --limit=5

# Get logs for specific job
gcloud logging read 'resource.type="aiplatform.googleapis.com/CustomJob" AND resource.labels.job_id="JOB_ID"' --limit=100
```

---

## Key Design Decisions

### Why SpatialDropout3D instead of regular Dropout?
Regular Dropout on ConvLSTM recurrent connections breaks CuDNN optimization, forcing CPU fallback (10x slower). SpatialDropout3D drops entire feature maps, maintaining GPU acceleration.

### Why AdamW instead of Adam + L2 regularization?
AdamW properly decouples weight decay from the adaptive learning rate. Standard L2 regularization with Adam is mathematically incorrect and provides weaker regularization.

### Why Kubeflow Pipelines?
- Parallel execution of experiments
- Reproducible DAG definition
- Integration with Vertex AI services
- Artifact lineage tracking

### Why download data to /tmp?
- NumPy's `np.load()` can't read from GCS paths
- TensorFlow's `tf.io.gfile` could work but would require rewriting data loading
- Local disk I/O is faster during training

### Why separate callbacks for TensorBoard vs Experiments?
- TensorBoard: Real-time curves, detailed histograms
- Experiments API: Structured comparison, programmatic querying, parameter tables

---

## Next Steps After Pipeline Completes

1. **Compare experiments** in Vertex AI Experiments console
2. **Identify best config** based on val_rmse_seconds
3. **Download best model** from GCS
4. **Run extended training** with winning configuration
5. **Iterate** with new hyperparameter ranges if needed
