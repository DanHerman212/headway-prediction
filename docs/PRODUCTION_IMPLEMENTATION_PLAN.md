# Production Implementation Plan: ST-ConvNet

Based on `docs/new_model_architecture.md`, we will refactor the codebase into a modular, object-oriented production system.

## 1. Directory Structure
```
headway_prediction/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration constants
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── dataset.py          # TF Dataset generation
│   ├── model.py            # ST-ConvNet architecture
│   └── trainer.py          # Training loop and evaluation
├── notebooks/
│   └── train_production.ipynb  # Driver notebook
└── requirements.txt
```

## 2. Class Design

### A. Configuration (`src/config.py`)
Centralized configuration to avoid magic numbers.
```python
@dataclass
class AppConfig:
    # Data Paths
    data_dir: str = "../data"
    headway_file: str = "headway_matrix_full.npy"
    schedule_file: str = "schedule_matrix_full.npy"
    
    # Model Params
    lookback_mins: int = 30
    forecast_mins: int = 15
    batch_size: int = 128
    scaler: float = 30.0
    
    # Training Params
    epochs: int = 30
    learning_rate: float = 1e-3
```

### B. Data Management (`src/data_loader.py`)
Handles raw file loading and normalization.
```python
class DataLoader:
    def __init__(self, config: AppConfig):
        self.config = config
        
    def load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Loads .npy files."""
        
    def preprocess(self, headway, schedule) -> Tuple[np.ndarray, np.ndarray]:
        """Normalizes and casts to float32."""
```

### C. Dataset Generation (`src/dataset.py`)
Encapsulates the complex `timeseries_dataset_from_array` logic.
```python
class TimeseriesGenerator:
    def __init__(self, config: AppConfig):
        self.config = config
        
    def create_train_val_datasets(self, headway, schedule):
        """Splits data and returns tf.data.Datasets."""
        
    def _make_dataset(self, headway, schedule, start_idx, end_idx):
        """Internal helper for creating a single dataset."""
```

### D. Model Architecture (`src/model.py`)
The ST-ConvNet implementation.
```python
class STConvNet(tf.keras.Model):
    def __init__(self, config: AppConfig, input_shape_h, input_shape_s):
        super().__init__()
        self.config = config
        # Define layers in __init__
        self.encoder_lstm_1 = layers.ConvLSTM2D(...)
        self.encoder_lstm_2 = layers.ConvLSTM2D(...)
        self.fusion_dense = layers.Dense(...)
        self.decoder_conv_1 = layers.Conv2D(...)
        # ...
        
    def call(self, inputs):
        # Define forward pass
        headway_in, schedule_in = inputs
        # ...
        return output
```

### E. Training & Evaluation (`src/trainer.py`)
Manages the training lifecycle.
```python
class ModelTrainer:
    def __init__(self, model, config: AppConfig):
        self.model = model
        self.config = config
        
    def train(self, train_ds, val_ds):
        """Runs model.fit with callbacks."""
        
    def evaluate(self, val_ds):
        """Runs evaluation metrics."""
        
    def save_model(self, path):
        """Saves the trained model."""
```

## 3. Implementation Steps
1.  **Setup**: Create `src/` directory and `__init__.py`.
2.  **Config & Data**: Implement `config.py`, `data_loader.py`, and `dataset.py`.
3.  **Model**: Implement `model.py` with the specific ConvLSTM + Broadcasting architecture.
4.  **Training**: Implement `trainer.py`.
5.  **Execution**: Create `train_production.ipynb` to wire it all together.
