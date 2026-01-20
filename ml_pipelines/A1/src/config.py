"""
Configuration for A1 Track Headway Prediction Model

Centralizes all hyperparameters, paths, and settings for:
- Data extraction and preprocessing
- Model architecture (Stacked GRU with multi-output)
- Training configuration
- Vertex AI Experiments and TensorBoard integration

Based on EDA findings:
- Lookback window: 20 events (from ACF analysis)
- Weak autocorrelation (5.37% max) - schedule-driven behavior
- Log transformation for outlier handling (170x skewness reduction)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class A1Config:
    """Configuration for A1 local track model."""
    
    # ============================================================================
    # Data Parameters
    # ============================================================================
    LOOKBACK_WINDOW: int = 20        # events (from ACF analysis)
    FORECAST_HORIZON: int = 1        # predict next 1 arrival (event-based)
    
    # Chronological data splits (no shuffling for time series)
    TRAIN_SPLIT: float = 0.6         # 60% for training
    VAL_SPLIT: float = 0.2           # 20% for validation
    TEST_SPLIT: float = 0.2          # 20% for testing
    
    # ============================================================================
    # Training Hyperparameters
    # ============================================================================
    BATCH_SIZE: int = 64
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    EARLY_STOPPING_PATIENCE: int = 10
    REDUCE_LR_PATIENCE: int = 3
    REDUCE_LR_FACTOR: float = 0.5
    MIN_LR: float = 1e-6
    
    # ============================================================================
    # Model Architecture - Stacked GRU
    # ============================================================================
    GRU_UNITS_1: int = 128           # first GRU layer
    GRU_UNITS_2: int = 64            # second GRU layer
    DROPOUT_RATE: float = 0.2
    DENSE_UNITS: int = 32            # shared dense layer before outputs
    
    # Multi-output heads
    NUM_ROUTES: int = 3              # A, C, E (one-hot encoded)
    REGRESSION_OUTPUT: int = 1       # headway prediction
    
    # ============================================================================
    # Optimizer & Loss Configuration
    # ============================================================================
    OPTIMIZER: str = "adam"
    REGRESSION_LOSS: str = "huber"
    CLASSIFICATION_LOSS: str = "categorical_crossentropy"
    
    # Loss weights (if needed for multi-output balancing)
    LOSS_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'route_output': 1.0,
        'headway_output': 1.0
    })
    
    # ============================================================================
    # Data Paths - BigQuery
    # ============================================================================
    BQ_PROJECT: str = "realtime-headway-prediction"
    BQ_DATASET: str = "headway_prediction"
    BQ_TABLE: str = "ml"
    BQ_LOCATION: str = "us-east1"
    
    # Local data paths
    DATA_DIR: str = "data/A1"
    RAW_DATA_CSV: str = "raw_data.csv"
    PREPROCESSED_NPY: str = "preprocessed_data.npy"
    SCALER_PARAMS_JSON: str = "scaler_params.json"
    
    # ============================================================================
    # Preprocessing Configuration
    # ============================================================================
    # Log transformation parameters
    LOG_OFFSET: float = 1.0          # log(headway + offset) to avoid log(0)
    
    # Feature names after preprocessing
    FEATURE_NAMES: list = field(default_factory=lambda: [
        'log_headway',
        'route_A', 'route_C', 'route_E',
        'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos'
    ])
    
    NUM_FEATURES: int = 8            # Total input features
    
    # ============================================================================
    # Vertex AI Configuration
    # ============================================================================
    EXPERIMENT_NAME: str = "a1-headway-prediction"
    GCS_BUCKET: str = "ml-pipelines-headway-prediction"
    TENSORBOARD_LOG_DIR: str = "gs://ml-pipelines-headway-prediction/tensorboard/A1"
    MODEL_ARTIFACTS_DIR: str = "gs://ml-pipelines-headway-prediction/models/A1"
    
    # ============================================================================
    # TensorBoard Tracking Configuration
    # ============================================================================
    # Scalars: loss, metrics, learning rate
    TRACK_SCALARS: bool = True
    
    # Histograms: weight distributions, gradients
    TRACK_HISTOGRAMS: bool = True
    HISTOGRAM_FREQ: int = 1          # Log every N epochs
    
    # Graph: model architecture
    TRACK_GRAPH: bool = True
    
    # HParams: hyperparameter comparison across runs
    TRACK_HPARAMS: bool = True
    
    # Profiling: GPU/CPU performance analysis
    TRACK_PROFILING: bool = True
    PROFILE_BATCH_RANGE: tuple = (10, 20)  # Profile batches 10-20
    
    # ============================================================================
    # Model Checkpoint Configuration
    # ============================================================================
    CHECKPOINT_DIR: str = "models/A1/checkpoints"
    BEST_MODEL_FILENAME: str = "best_model.keras"
    MONITOR_METRIC: str = "val_loss"
    CHECKPOINT_MODE: str = "min"     # 'min' for loss, 'max' for accuracy
    
    # ============================================================================
    # Evaluation Configuration
    # ============================================================================
    # Baseline models for comparison
    BASELINE_TYPES: list = field(default_factory=lambda: [
        'persistence',               # next headway = last headway
        'mean_by_hour',             # mean headway grouped by hour
        'mean_by_hour_route'        # mean headway grouped by hour + route
    ])
    
    # ============================================================================
    # Helper Properties
    # ============================================================================
    @property
    def raw_data_path(self) -> str:
        """Full path to raw CSV data."""
        return os.path.join(self.DATA_DIR, self.RAW_DATA_CSV)
    
    @property
    def preprocessed_data_path(self) -> str:
        """Full path to preprocessed numpy data."""
        return os.path.join(self.DATA_DIR, self.PREPROCESSED_NPY)
    
    @property
    def scaler_params_path(self) -> str:
        """Full path to scaler parameters JSON."""
        return os.path.join(self.DATA_DIR, self.SCALER_PARAMS_JSON)
    
    @property
    def checkpoint_path(self) -> str:
        """Full path to best model checkpoint."""
        return os.path.join(self.CHECKPOINT_DIR, self.BEST_MODEL_FILENAME)
    
    @property
    def hparams_dict(self) -> Dict[str, Any]:
        """Dictionary of hyperparameters for TensorBoard HParams logging."""
        return {
            'lookback_window': self.LOOKBACK_WINDOW,
            'forecast_horizon': self.FORECAST_HORIZON,
            'batch_size': self.BATCH_SIZE,
            'learning_rate': self.LEARNING_RATE,
            'gru_units_1': self.GRU_UNITS_1,
            'gru_units_2': self.GRU_UNITS_2,
            'dropout_rate': self.DROPOUT_RATE,
            'dense_units': self.DENSE_UNITS,
            'optimizer': self.OPTIMIZER,
            'regression_loss': self.REGRESSION_LOSS,
            'classification_loss': self.CLASSIFICATION_LOSS,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# Default configuration instance
config = A1Config()
