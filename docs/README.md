# ML Pipeline for Headway Prediction

A production-ready machine learning pipeline with comprehensive Vertex AI Experiments and TensorBoard integration.

## Features

### 🔬 Experiment Tracking
- **Full Vertex AI Experiments Integration**: Every run is tracked with hyperparameters, metrics, and metadata
- **Complete TensorBoard Logging**: 
  - Scalars (loss, metrics, learning rate)
  - Histograms (weight distributions, gradients)
  - Graphs (model architecture visualization)
  - HParams (hyperparameter comparison)
  - Profiling (GPU/CPU performance analysis)
  - Custom visualizations (predictions, attention maps)

### 📊 Data Pipeline
- **BigQuery ETL**: Direct data extraction from BigQuery with SQL support
- **Efficient Processing**: Train/val/test splitting with proper scaling (fit on train only)
- **TensorFlow Datasets**: Optimized tf.data.Dataset pipelines with prefetching and batching

### 🏗️ Modular Architecture
- **Configuration Management**: Centralized configs for models and tracking
- **Model Framework**: Base classes and builders for custom architectures
- **Training Framework**: Trainer class with automatic callback integration
- **Evaluation Suite**: Comprehensive metrics and visualizations

### 🚀 Vertex AI Pipelines (KFP)
- End-to-end orchestration with Kubeflow Pipelines
- Automatic experiment tracking across pipeline steps
- Cloud-native execution on Vertex AI

## Project Structure

```
ml_pipelines/
├── __init__.py                 # Package initialization
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── config/                     # Configuration management
│   ├── model_config.py         # Model hyperparameters and settings
│   └── tracking_config.py      # Experiment tracking configuration
│
├── data/                       # Data pipeline
│   ├── bigquery_etl.py         # BigQuery data extraction and transformation
│   └── dataset_generator.py   # TensorFlow dataset creation
│
├── models/                     # Model architectures
│   ├── base_model.py           # Abstract base class for models
│   └── model_builder.py        # Model compilation utilities
│
├── training/                   # Training framework
│   └── trainer.py              # Training orchestration with tracking
│
├── evaluation/                 # Model evaluation
│   ├── metrics.py              # Custom metrics (RMSE, R², etc.)
│   └── evaluator.py            # Evaluation and visualization
│
├── tracking/                   # Experiment tracking (⭐ KEY FEATURE)
│   ├── tracker.py              # ExperimentTracker class
│   └── callbacks.py            # Keras callbacks for auto-logging
│
└── pipelines/                  # Vertex AI Pipelines
    └── training_pipeline.py    # KFP pipeline definition
```

## Quick Start

### 1. Installation

```bash
cd ml_pipelines
pip install -r requirements.txt
```

### 2. Basic Training Script

```python
from ml_pipelines.config import ModelConfig, TrackingConfig
from ml_pipelines.tracking import ExperimentTracker
from ml_pipelines.data.bigquery_etl import BigQueryETL
from ml_pipelines.training import Trainer
from ml_pipelines.evaluation import Evaluator
from ml_pipelines.evaluation.metrics import rmse_seconds, r_squared

# 1. Configure model
model_config = ModelConfig(
    model_name="my_model",
    task_type="regression",
    lookback_steps=30,
    forecast_steps=15,
    batch_size=128,
    epochs=100,
    learning_rate=5e-4,
    # BigQuery configuration
    bq_project="my-project",
    bq_dataset="ml_data",
    bq_table="training_data",
)

# 2. Configure tracking (⭐ THIS IS THE KEY)
tracking_config = TrackingConfig.create_from_model_config(
    model_config=model_config,
    experiment_name="my-experiment",
    run_name="run-001",
    vertex_project="my-project",
    vertex_location="us-east1",
    # Enable all tracking features
    scalars=True,
    histograms=True,
    graphs=True,
    hparams=True,
    profiling=False,  # Enable only for performance debugging
)

# 3. Initialize experiment tracker
tracker = ExperimentTracker(tracking_config)

# 4. Load data from BigQuery
etl = BigQueryETL(
    project_id=model_config.bq_project,
    dataset_id=model_config.bq_dataset,
    table_id=model_config.bq_table
)

df = etl.load_data()
train_df, val_df, test_df, scaler = etl.split_and_scale(df)

# 5. Create datasets
# TODO: Convert to tf.data.Dataset based on your model requirements

# 6. Build model
# TODO: Import and create your model
# from ml_pipelines.models import YourModel
# model = YourModel(model_config).create()

# 7. Create trainer with tracker integration
trainer = Trainer(
    model=model,
    config=model_config,
    tracker=tracker,  # ⭐ Tracker automatically logs everything
)

# 8. Compile with custom metrics
trainer.compile(metrics=[rmse_seconds, r_squared])

# 9. Train (tracker callbacks are added automatically)
history = trainer.fit(
    train_dataset=train_ds,
    val_dataset=val_ds
)

# 10. Evaluate
evaluator = Evaluator(model, model_config, scaler)
test_metrics = evaluator.evaluate(test_ds)

# 11. Close tracker
tracker.close()

# View results:
# - TensorBoard: tensorboard --logdir=tensorboard_logs/my-experiment/run-001
# - Vertex AI: https://console.cloud.google.com/vertex-ai/experiments
```

## Experiment Tracking Details

### What Gets Logged Automatically

When you use `ExperimentTracker`, the following data is logged to both TensorBoard and Vertex AI:

1. **Scalars** (every epoch):
   - Training loss, validation loss
   - All custom metrics (RMSE, R², MAE, etc.)
   - Learning rate

2. **Histograms** (configurable frequency):
   - Weight distributions for all layers
   - Gradient distributions (optional)

3. **Model Graph** (once at start):
   - Architecture visualization
   - Layer connections
   - Tensor shapes

4. **Hyperparameters** (at start):
   - All model configuration parameters
   - Logged to both TensorBoard HParams plugin and Vertex AI

5. **Final Metrics** (at end):
   - Best validation metrics
   - Associated with hyperparameters for comparison

### Viewing Results

#### TensorBoard (Local/Cloud)
```bash
# Local TensorBoard
tensorboard --logdir=tensorboard_logs/my-experiment

# Cloud TensorBoard (if using GCS)
tensorboard --logdir=gs://my-bucket/tensorboard/my-experiment
```

**TensorBoard Tabs Available:**
- **Scalars**: Loss and metric curves
- **Graphs**: Model architecture
- **Distributions**: Weight/gradient distributions over time
- **Histograms**: Weight/gradient histograms
- **HParams**: Hyperparameter comparison across runs
- **Profile**: GPU/CPU performance analysis (if enabled)

#### Vertex AI Experiments
Navigate to: https://console.cloud.google.com/vertex-ai/experiments

**Features:**
- Compare runs side-by-side
- Filter and sort by metrics
- Download experiment data
- Link to TensorBoard instances

## Creating Custom Models

```python
from ml_pipelines.models.base_model import BaseModel
import tensorflow as tf

class MyCustomModel(BaseModel):
    """Your custom model architecture."""
    
    def build(self) -> tf.keras.Model:
        """Implement your architecture here."""
        inputs = tf.keras.Input(shape=self.config.input_shape)
        
        # Add your layers
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_custom_model")
        return model

# Usage
model = MyCustomModel(model_config).create()
```

## Vertex AI Pipeline Deployment

```bash
# 1. Compile pipeline
python -m ml_pipelines.pipelines.training_pipeline --compile

# 2. Submit to Vertex AI
python -m ml_pipelines.pipelines.training_pipeline --submit \
    --experiment-name my-experiment \
    --run-name run-001
```

## Configuration via YAML

```yaml
# model_config.yaml
model_name: "my_model"
task_type: "regression"
lookback_steps: 30
forecast_steps: 15
batch_size: 128
epochs: 100
learning_rate: 0.0005
bq_project: "my-project"
bq_dataset: "ml_data"
bq_table: "training_data"

# tracking_config.yaml
experiment_name: "my-experiment"
run_name: "run-001"
scalars: true
histograms: true
histogram_freq: 1
graphs: true
hparams: true
profiling: false
vertex_project: "my-project"
```

Load configs:
```python
model_config = ModelConfig.from_yaml("model_config.yaml")
tracking_config = TrackingConfig.from_yaml("tracking_config.yaml")
```

## Next Steps

1. **Define your prediction task**: Update `ModelConfig` with task-specific parameters
2. **Build ETL pipeline**: Implement data extraction from your BigQuery tables
3. **Create model architecture**: Subclass `BaseModel` or use existing architectures
4. **Configure experiments**: Set up your Vertex AI project and TensorBoard instance
5. **Run experiments**: Train models with full tracking enabled

## Support

For issues or questions about this pipeline framework, refer to the implementation in `archive/backup/src` which this is based on.
