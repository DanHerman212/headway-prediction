# Vertex AI Experiment Tracking Integration

## 🎯 Architecture Overview

Your existing tracking framework is **production-ready** and integrates seamlessly with the modular architecture:

```
┌─────────────┐
│   .env      │  Single Source of Truth
│  Config     │  (All params in one place)
└──────┬──────┘
       │
       ├──────────────┬──────────────┬──────────────┐
       │              │              │              │
┌──────▼──────┐ ┌────▼─────┐  ┌────▼─────┐  ┌─────▼──────┐
│ ModelConfig │ │GRUModel  │  │ Trainer  │  │  Tracker   │
│             │ │ (arch)   │  │  (data   │  │  (Vertex+  │
│  • GRU units│ │          │  │   +train)│  │   TBoard)  │
│  • Losses   │ └──────────┘  └──────────┘  └────────────┘
│  • Tracking │
└─────────────┘
```

---

## 🔌 Integration Flow

### Step 1: Configuration (Already Done! ✅)

```python
# .env defines everything
GRU_UNITS=128,64
EXPERIMENT_NAME=headway-prediction-experiments
USE_VERTEX_EXPERIMENTS=true
TRACK_HISTOGRAMS=true
```

### Step 2: Create Model with Config

```python
from models.gru_model import StackedGRUModel
from config import ModelConfig

# Load config from .env
config = ModelConfig.from_env()

# Model uses config for architecture
model_builder = StackedGRUModel(config)
model = model_builder.create()
```

### Step 3: Create Tracker from Config

```python
from tracking import ExperimentTracker, TrackingConfig

# TrackingConfig extracts tracking params from ModelConfig
tracking_config = TrackingConfig.create_from_model_config(
    model_config=config,
    experiment_name=config.experiment_name,
    run_name=f"{config.model_name}-{timestamp}",
    vertex_project=config.bq_project,
    vertex_location=config.vertex_location,
    use_vertex_experiments=config.use_vertex_experiments,
    histograms=config.track_histograms,
    histogram_freq=config.histogram_freq,
    profiling=config.track_profiling,
    profile_batch_range=config.profile_batch_range
)

# Initialize tracker
tracker = ExperimentTracker(tracking_config)
```

### Step 4: Train with Automatic Tracking

```python
from training import Trainer

# Trainer handles data and training
trainer = Trainer(config)
trainer.load_data('data/X.csv')

# Compile model with tracker's help
model.compile(
    optimizer=tf.keras.optimizers.Adam(config.learning_rate),
    loss={
        'headway': config.regression_loss,
        'route': config.classification_loss
    },
    loss_weights=config.loss_weights,
    metrics={
        'headway': ['mae'],
        'route': ['accuracy']
    }
)

# Log model graph to TensorBoard + Vertex AI
tracker.log_graph(model)

# Train with tracker's callbacks
history = trainer.train(
    model=model,
    epochs=config.epochs,
    callbacks=tracker.keras_callbacks()  # Auto-logs everything!
)

# Close tracker (logs final metrics)
tracker.close()
```

---

## 📊 What Gets Tracked Automatically

Your `ExperimentTracker` with callbacks logs:

### TensorBoard + Vertex AI:
- ✅ **Scalars** (every epoch):
  - `epoch/training/loss`, `epoch/training/headway_loss`, `epoch/training/route_loss`
  - `epoch/validation/loss`, `epoch/validation/headway_mae`, `epoch/validation/route_accuracy`
  - `epoch/learning_rate`

- ✅ **Histograms** (every N epochs):
  - Weight distributions for all layers (GRU kernels, biases)
  - Gradient distributions (optional, for debugging vanishing/exploding gradients)

- ✅ **HParams** (experiment start):
  - All hyperparameters from ModelConfig
  - Enables TensorBoard HParams tab for comparison

- ✅ **Model Graph** (once):
  - Full model architecture visualization

- ✅ **Profiling** (optional):
  - GPU/CPU utilization
  - Memory usage

### Vertex AI Only:
- ✅ Experiment comparison UI
- ✅ Metric tracking across runs
- ✅ Artifact storage
- ✅ TensorBoard instance integration

---

## 🏗️ Complete Training Script

```python
"""
training/sample.py
Complete training script with Vertex AI tracking
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU for small tests

from datetime import datetime
from config import ModelConfig, TrackingConfig
from models.gru_model import StackedGRUModel
from training import Trainer
from tracking import ExperimentTracker

if __name__ == "__main__":
    # 1. Load configuration (single source of truth)
    print("Loading configuration...")
    config = ModelConfig.from_env()
    
    # 2. Create tracking configuration
    print("Initializing experiment tracking...")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tracking_config = TrackingConfig.create_from_model_config(
        model_config=config,
        experiment_name=config.experiment_name,
        run_name=f"{config.model_name}-{timestamp}",
        vertex_project=config.bq_project,
        vertex_location=config.vertex_location,
        use_vertex_experiments=config.use_vertex_experiments,
        histograms=config.track_histograms,
        histogram_freq=config.histogram_freq,
        profiling=config.track_profiling,
        profile_batch_range=config.profile_batch_range
    )
    
    # 3. Initialize tracker
    tracker = ExperimentTracker(tracking_config)
    
    try:
        # 4. Build model
        print(f"Building {config.model_type} model...")
        model_builder = StackedGRUModel(config)
        model = model_builder.create()
        
        # 5. Compile model with multi-output losses
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss={
                'headway': config.regression_loss,
                'route': config.classification_loss
            },
            loss_weights=config.loss_weights,
            metrics={
                'headway': ['mae'],
                'route': ['accuracy']
            }
        )
        
        # 6. Log model architecture
        tracker.log_graph(model)
        
        # 7. Load data
        print("Loading and creating datasets...")
        trainer = Trainer(config)
        trainer.load_data('data/X.csv')
        
        # 8. Train with automatic tracking
        print("Starting training...")
        history = trainer.train(
            model=model,
            epochs=config.epochs,
            callbacks=tracker.keras_callbacks()
        )
        
        # 9. Log final metrics
        print("Training complete!")
        tracker.log_text(
            "experiment/summary",
            f"Final metrics:\\n{history.history}",
            step=config.epochs
        )
        
    finally:
        # Always close tracker
        tracker.close()
        print(f"\\n✓ Experiment logged to Vertex AI: {config.experiment_name}/{tracking_config.run_name}")
        print(f"✓ View TensorBoard: {tracking_config.get_tensorboard_command()}")
```

---

## 💡 Key Benefits

### ✅ Single Source of Truth
- All params in `.env` → `ModelConfig`
- No magic numbers scattered in code
- Easy to version control experiments

### ✅ Automatic Tracking
- No manual logging code needed
- Callbacks handle everything
- Works with Keras `.fit()` out of the box

### ✅ Modular & Testable
- `models/` = architecture (no tracking code)
- `training/` = data + training logic
- `tracking/` = experiment tracking (reusable)

### ✅ Production-Ready
- Vertex AI integration for team collaboration
- TensorBoard for deep analysis
- Full reproducibility

---

## 🚀 Next Steps

1. **Create `models/gru_model.py`** with stacked GRU architecture
2. **Update `training/train.py`** to accept model and compile with multi-output losses
3. **Add custom MAE-in-seconds metric** (converts from log-space to seconds)
4. **Test in notebook** to verify tracking works
5. **Create `training/sample.py`** as shown above

Ready to implement! 🎯
