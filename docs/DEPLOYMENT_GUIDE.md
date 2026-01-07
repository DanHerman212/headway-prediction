# Deployment Guide: Vertex AI Pipeline

Step-by-step instructions for training the headway prediction model on Google Cloud Vertex AI.

---

## Prerequisites

### 1. Local Environment

```bash
# Python 3.10+
python3 --version

# Google Cloud SDK
gcloud --version

# Docker
docker --version
```

### 2. GCP Authentication

```bash
# Login to GCP
gcloud auth login

# Set project
gcloud config set project time-series-478616

# Configure Docker for GCR
gcloud auth configure-docker gcr.io
```

### 3. Verify Data on GCS

```bash
# Check data exists
gsutil ls gs://st-convnet-training-configuration/data/

# Expected files:
#   headway_matrix_full.npy
#   schedule_matrix_full.npy
```

---

## Step 1: Verify Local Setup

Before deploying, ensure the code works locally.

### 1.1 Test Model Build

```bash
cd /Users/danherman/Desktop/headway-prediction

python3 -c "
from src.config import Config
from src.models.baseline_convlstm import HeadwayConvLSTM

config = Config()
model = HeadwayConvLSTM(config).build_model()
print(f'Model: {model.name}')
print(f'Params: {model.count_params():,}')
"
```

Expected output:
```
Model: BaselineConvLSTM
Params: 187,425
```

### 1.2 Test Data Loading

```bash
python3 -c "
from src.config import Config
from src.data.dataset import SubwayDataGenerator

config = Config()
gen = SubwayDataGenerator(config)
gen.load_data(normalize=True)

ds = gen.make_dataset(start_index=0, end_index=100)
for batch in ds.take(1):
    inputs, targets = batch
    print(f'Headway: {inputs[\"headway_input\"].shape}')
    print(f'Schedule: {inputs[\"schedule_input\"].shape}')
    print(f'Target: {targets.shape}')
"
```

Expected output:
```
Headway: (32, 30, 66, 2, 1)
Schedule: (32, 15, 2, 1)
Target: (32, 15, 66, 2, 1)
```

### 1.3 Test Full Pipeline (Quick)

```bash
python3 -c "
from src.config import Config
from src.data.dataset import SubwayDataGenerator
from src.models.baseline_convlstm import HeadwayConvLSTM
from src.training.trainer import Trainer

# Setup
config = Config()
config.EPOCHS = 1  # Quick test

# Data
gen = SubwayDataGenerator(config)
gen.load_data(normalize=True)
train_ds = gen.make_dataset(0, 100, shuffle=True)
val_ds = gen.make_dataset(100, 200, shuffle=False)

# Model
model = HeadwayConvLSTM(config).build_model()

# Train
trainer = Trainer(model, config)
trainer.compile_model()
history = trainer.fit(train_ds, val_ds, patience=1)

print('Pipeline test PASSED')
"
```

---

## Step 2: Build Docker Container

### 2.1 Review Dockerfile

The Dockerfile should use the TensorFlow GPU base image:

```dockerfile
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg

ENTRYPOINT ["python"]
CMD ["-m", "src.experiments.run_experiment", "--help"]
```

### 2.2 Build Image

```bash
cd /Users/danherman/Desktop/headway-prediction

docker build -t gcr.io/time-series-478616/headway-trainer:latest .
```

### 2.3 Test Container Locally (Optional)

```bash
docker run --rm gcr.io/time-series-478616/headway-trainer:latest \
    -c "from src.config import Config; print(Config())"
```

### 2.4 Push to Container Registry

```bash
docker push gcr.io/time-series-478616/headway-trainer:latest
```

---

## Step 3: Upload Data to GCS

If data is not already on GCS:

```bash
# Upload data files
gsutil -m cp data/headway_matrix_full.npy gs://st-convnet-training-configuration/data/
gsutil -m cp data/schedule_matrix_full.npy gs://st-convnet-training-configuration/data/
gsutil -m cp data/a_line_station_distances.csv gs://st-convnet-training-configuration/data/
```

---

## Step 4: Submit Training Pipeline

### Compile Pipeline (Optional - for inspection)

```bash
python -m src.experiments.pipeline --compile

# Outputs: headway_pipeline.json
```

### Submit to Vertex AI

```bash
python -m src.experiments.pipeline --submit --run_name baseline-001
```

With custom parameters:
```bash
python -m src.experiments.pipeline --submit \
    --run_name experiment-002 \
    --epochs 100 \
    --batch_size 32 \
    --filters 32
```

### Pipeline Steps

The pipeline runs three steps in sequence:

1. **Data Component** - Loads data using `SubwayDataGenerator`
2. **Training Component** - Trains model using `HeadwayConvLSTM` + `Trainer` with A100 GPU
3. **Evaluation Component** - Evaluates model using `Evaluator`

All steps log to the same TensorBoard directory for unified visualization.

---

## Step 5: Monitor Training

### 5.1 View Job Status

```bash
# List recent jobs
gcloud ai custom-jobs list \
    --project=time-series-478616 \
    --region=us-east1 \
    --limit=5

# Get job details
gcloud ai custom-jobs describe JOB_ID \
    --project=time-series-478616 \
    --region=us-east1
```

### 5.2 View Logs

```bash
gcloud ai custom-jobs stream-logs JOB_ID \
    --project=time-series-478616 \
    --region=us-east1
```

### 5.3 TensorBoard

If using Vertex AI TensorBoard:

```bash
# Create TensorBoard instance (one-time)
gcloud ai tensorboards create \
    --display-name="headway-experiments" \
    --project=time-series-478616 \
    --region=us-east1

# View TensorBoard URL in console
# https://console.cloud.google.com/vertex-ai/experiments/tensorboard-instances
```

Or run TensorBoard locally pointing to GCS:

```bash
tensorboard --logdir=gs://st-convnet-training-configuration/tensorboard/
```

---

## Step 6: Retrieve Results

### 6.1 Download Model Checkpoint

```bash
gsutil cp gs://st-convnet-training-configuration/runs/exp_01_baseline/best_model.keras ./models/
```

### 6.2 Download Training Metrics

```bash
gsutil -m cp -r gs://st-convnet-training-configuration/runs/exp_01_baseline/tensorboard/ ./outputs/
```

### 6.3 Evaluate Locally

```python
from tensorflow import keras
from src.evaluator import Evaluator
from src.config import Config

model = keras.models.load_model("models/best_model.keras")
evaluator = Evaluator(Config())
evaluator.evaluate_predictions(model, test_ds)
```

---

## Experiment Configurations

The following experiments are pre-defined in `experiment_config.py`:

| ID | Name | Description |
|----|------|-------------|
| 1 | baseline | No regularization (paper-faithful) |
| 2 | dropout_only | SpatialDropout3D=0.2 |
| 3 | weight_decay_only | AdamW weight_decay=1e-4 |
| 4 | combined | Dropout + Weight Decay |

Run specific experiments:

```bash
# Single experiment
python3 -m src.experiments.vertex_pipeline --experiments 1

# Multiple experiments
python3 -m src.experiments.vertex_pipeline --experiments 1 2 3 4
```

---

## Troubleshooting

### Container Build Fails

```bash
# Check Docker is running
docker info

# Build with verbose output
docker build --progress=plain -t gcr.io/time-series-478616/headway-trainer:latest .
```

### Push Fails (Authentication)

```bash
# Re-authenticate
gcloud auth configure-docker gcr.io --quiet
docker push gcr.io/time-series-478616/headway-trainer:latest
```

### Job Fails Immediately

1. Check container runs locally:
   ```bash
   docker run --rm gcr.io/time-series-478616/headway-trainer:latest -c "print('OK')"
   ```

2. Check data exists on GCS:
   ```bash
   gsutil ls gs://st-convnet-training-configuration/data/
   ```

3. Check service account permissions

### Out of Memory

Reduce batch size in `Config`:
```python
config.BATCH_SIZE = 16  # or smaller
```

---

## Cost Estimation

| Resource | Cost/Hour | Typical Run |
|----------|-----------|-------------|
| a2-highgpu-1g (A100) | ~$3.67 | ~$7-15 per experiment |
| n1-standard-8 (CPU) | ~$0.38 | ~$2-4 per experiment |

Baseline training (100 epochs, early stopping at ~50) typically takes 2-4 hours on A100.
