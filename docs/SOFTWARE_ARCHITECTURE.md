# Software Architecture

## Overview

This document describes the software architecture for the headway prediction system. The codebase follows a modular design with clear separation of concerns, enabling easy experimentation while maintaining production quality.

## Directory Structure

```
src/
├── __init__.py
├── config.py           # Central configuration (single source of truth)
├── metrics.py          # Custom TensorFlow metrics
├── evaluator.py        # Post-training analysis and visualization
├── data/
│   └── dataset.py      # Data loading and tf.data pipeline
├── models/
│   ├── baseline_convlstm.py  # Paper-faithful baseline (ground truth)
│   ├── st_convnet.py         # Original ConvLSTM with regularization
│   ├── st_convnet_v2.py      # Improved architecture
│   └── st_convnet_paper.py   # Alternative paper implementation
├── training/
│   └── trainer.py      # Compilation and training loop
├── tracking/
│   ├── config.py       # TrackerConfig dataclass
│   ├── tracker.py      # TensorBoard logging interface
│   └── callbacks.py    # Keras callbacks for tracking
├── visualizations/
│   └── spatiotemporal.py  # Headway heatmap visualizations
└── experiments/
    ├── experiment_config.py  # Experiment definitions
    ├── run_experiment.py     # Single experiment runner
    ├── kfp_pipeline.py       # Kubeflow pipeline definition
    └── vertex_pipeline.py    # Vertex AI job submission
```

---

## Core Components

### 1. config.py - Configuration

**Purpose:** Single source of truth for all hyperparameters.

```python
from src.config import Config

config = Config()
config.FILTERS = 32       # Paper default
config.BATCH_SIZE = 32    # Paper default
config.EPOCHS = 100       # Paper default
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LOOKBACK_MINS` | 30 | Historical window (paper) |
| `FORECAST_MINS` | 15 | Prediction horizon (paper) |
| `BATCH_SIZE` | 32 | Training batch size (paper) |
| `EPOCHS` | 100 | Max training epochs (paper) |
| `EARLY_STOPPING_PATIENCE` | 50 | Epochs before stopping (paper) |
| `FILTERS` | 32 | ConvLSTM filters (paper) |
| `KERNEL_SIZE` | (3, 3) | Convolution kernel |
| `NUM_STATIONS` | 66 | A-line station count |

---

### 2. src/data/dataset.py - Data Pipeline

**Purpose:** Load data and create efficient tf.data.Dataset pipelines.

```python
from src.config import Config
from src.data.dataset import SubwayDataGenerator

config = Config()
gen = SubwayDataGenerator(config)
gen.load_data(normalize=True, max_headway=30.0)

train_ds = gen.make_dataset(start_index=0, end_index=train_end, shuffle=True)
val_ds = gen.make_dataset(start_index=train_end, end_index=None, shuffle=False)
```

**Data Shapes:**
- Input `headway_input`: `(batch, 30, 66, 2, 1)` - 30 min history, 66 stations, 2 directions
- Input `schedule_input`: `(batch, 15, 2, 1)` - 15 min future terminal schedule
- Target: `(batch, 15, 66, 2, 1)` - 15 min future headways

---

### 3. src/models/ - Model Architectures

**Purpose:** Define model architectures. Models only implement `build_model()` - no compilation.

**Available Models:**

| File | Class | Description |
|------|-------|-------------|
| `baseline_convlstm.py` | `HeadwayConvLSTM` | Paper-faithful baseline (ground truth) |
| `st_convnet.py` | `HeadwayConvLSTM` | With spatial dropout regularization |
| `st_convnet_v2.py` | `HeadwayConvLSTM` | Improved bottleneck architecture |

**Usage Pattern:**
```python
from src.config import Config
from src.models.baseline_convlstm import HeadwayConvLSTM

config = Config()
builder = HeadwayConvLSTM(config)
model = builder.build_model()  # Returns uncompiled Keras model
```

**Architecture (Paper - arXiv:2510.03121):**
```
Encoder:  ConvLSTM(32) → BN → ConvLSTM(32) → BN
Bridge:   Repeat state × forecast_steps
Fusion:   Concatenate(state, broadcast_schedule)
Decoder:  ConvLSTM(32) → BN
Output:   Conv3D(1, sigmoid)
```

---

### 4. src/training/trainer.py - Training Loop

**Purpose:** Handle model compilation and training with callbacks.

```python
from src.training.trainer import Trainer

trainer = Trainer(model, config, checkpoint_dir="models/baseline")
trainer.compile_model()

# Optional: Add tracking callbacks
from src.tracking import Tracker, TrackerConfig
tracker = Tracker(TrackerConfig(...))

history = trainer.fit(
    train_ds, 
    val_ds,
    extra_callbacks=tracker.keras_callbacks()
)
```

**Features:**
- Compiles with Adam optimizer, MSE loss
- Metrics: `rmse_seconds`, `r_squared`
- Built-in callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Accepts extra callbacks for tracking integration

---

### 5. src/metrics.py - Custom Metrics

**Purpose:** Production-relevant metrics in real units.

| Metric | Description |
|--------|-------------|
| `rmse_seconds(y_true, y_pred)` | RMSE in seconds (assumes 30 min = 1.0) |
| `mae_seconds(y_true, y_pred)` | MAE in seconds |
| `r_squared(y_true, y_pred)` | Coefficient of determination |

---

### 6. src/evaluator.py - Post-Training Analysis

**Purpose:** Generate paper-style visualizations and analysis.

```python
from src.evaluator import Evaluator

evaluator = Evaluator(config)
evaluator.plot_training_curves(history, save_path="results/curves.png")
evaluator.evaluate_predictions(model, test_ds)
```

---

### 7. src/tracking/ - TensorBoard Integration

**Purpose:** Comprehensive TensorBoard logging for all tabs.

**Components:**

| File | Purpose |
|------|---------|
| `config.py` | `TrackerConfig` - Configuration dataclass |
| `tracker.py` | `Tracker` - Main logging interface |
| `callbacks.py` | Keras callbacks for automatic logging |

**Usage:**
```python
from src.tracking import Tracker, TrackerConfig

tracker_config = TrackerConfig(
    experiment_name="baseline-experiments",
    run_name="exp01-paper-baseline",
    log_dir="gs://bucket/tensorboard/exp01",
    histograms=True,
    hparams=True,
    hparams_dict={"filters": 32, "batch_size": 32}
)

tracker = Tracker(tracker_config)

# Pass to Trainer
history = trainer.fit(
    train_ds, val_ds,
    extra_callbacks=tracker.keras_callbacks()
)

tracker.close()
```

**TensorBoard Tabs Supported:**
- **Scalars:** Loss, metrics, learning rate per epoch
- **Histograms:** Weight distributions (debugging vanishing/exploding gradients)
- **Graphs:** Model architecture visualization
- **HParams:** Hyperparameter comparison across runs
- **Images:** Spatiotemporal heatmaps (via `SpatiotemporalCallback`)

---

### 8. src/visualizations/ - Custom Visualizations

**Purpose:** Domain-specific visualizations for headway prediction.

```python
from src.visualizations import SpatiotemporalCallback

viz_callback = SpatiotemporalCallback(
    tracker=tracker,
    validation_data=val_ds,
    freq=5,  # Every 5 epochs
)

# Add to training
trainer.fit(train_ds, val_ds, extra_callbacks=[viz_callback])
```

---

### 9. src/experiments/ - Pipeline Orchestration

**Purpose:** KFP pipeline that orchestrates the existing modules.

| File | Purpose |
|------|---------|
| `pipeline.py` | KFP pipeline definition with 3 steps |

**Pipeline Steps:**

```
Step 1: data_component      → wraps SubwayDataGenerator
Step 2: training_component  → wraps HeadwayConvLSTM + Trainer + Tracker
Step 3: evaluation_component → wraps Evaluator + Tracker
```

**Usage:**
```bash
# Compile pipeline
python -m src.experiments.pipeline --compile

# Submit to Vertex AI
python -m src.experiments.pipeline --submit --run_name baseline-001
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        config.py                                 │
│                  (Single Source of Truth)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   data/         │  │   models/       │  │   tracking/     │
│   dataset.py    │  │   *.py          │  │   tracker.py    │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         │                    ▼                    │
         │           ┌─────────────────┐           │
         └──────────▶│   training/     │◀──────────┘
                     │   trainer.py    │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │   evaluator.py  │
                     │   metrics.py    │
                     └─────────────────┘
```

---

## Design Principles

1. **Config is the single source of truth** - All hyperparameters flow from Config
2. **Models only define architecture** - No compilation in model classes
3. **Trainer owns the training loop** - Compilation, callbacks, checkpointing
4. **Tracking is pluggable** - Pass callbacks to Trainer, don't modify core logic
5. **Experiments are reproducible** - Config + data version = reproducible run
