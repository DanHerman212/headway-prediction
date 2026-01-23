"""
Model Configuration

Centralized configuration for model hyperparameters, architecture,
data paths, and training settings. Extensible for new prediction tasks.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml


@dataclass
class ModelConfig:
    """
    Configuration for model architecture and training.
    
    This serves as the single source of truth for all model hyperparameters.
    Override defaults for specific prediction tasks.
    
    Attributes:
        # Model Identity
        model_name: Name/identifier for this model configuration
        task_type: Type of prediction task (e.g., "regression", "classification")
        
        # Temporal Parameters
        lookback_steps: Number of historical timesteps to use as input
        forecast_steps: Number of future timesteps to predict
        time_resolution_mins: Time resolution in minutes (e.g., 1 for per-minute)
        
        # Training Parameters
        batch_size: Training batch size
        epochs: Maximum number of training epochs
        learning_rate: Initial learning rate for optimizer
        early_stopping_patience: Epochs to wait before early stopping
        reduce_lr_patience: Epochs to wait before reducing LR on plateau
        
        # Architecture Parameters
        filters: Number of convolutional filters (if applicable)
        hidden_units: Hidden layer units for dense layers
        dropout_rate: Dropout rate for regularization
        
        # Data Configuration
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        
        # Scaling/Normalization
        scaling_method: Method for data scaling ("minmax", "standard", "robust", "none")
        target_range: Range for scaling (e.g., (0, 1) for MinMaxScaler)
        
        # Data Paths (support both local and GCS)
        data_dir: Local data directory
        data_gcs_path: GCS path for data (if using cloud storage)
        
        # Output Paths
        model_output_dir: Directory for saved models
        checkpoint_dir: Directory for training checkpoints
        
        # BigQuery Configuration (for ETL)
        bq_project: BigQuery project ID
        bq_dataset: BigQuery dataset name
        bq_table: BigQuery table name for training data
        
        # Additional parameters
        custom_params: Dictionary for task-specific parameters
    """
    
    # =========================================================================
    # Model Identity
    # =========================================================================
    model_name: str = "baseline_model"
    task_type: str = "regression"
    description: Optional[str] = None
    
    # =========================================================================
    # Temporal Parameters
    # =========================================================================
    lookback_steps: int = 30
    forecast_steps: int = 15
    time_resolution_mins: int = 1
    
    # =========================================================================
    # Data Extraction
    # =========================================================================
    track: str = "A1"
    route_ids: tuple = ("A", "C", "E")
    
    # =========================================================================
    # Training Parameters
    # =========================================================================
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 5e-4
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 3
    min_learning_rate: float = 1e-6
    
    # Optimizer settings
    optimizer: str = "adam"
    gradient_clip_norm: Optional[float] = 1.0
    
    # Loss function
    loss: str = "huber"
    
    # =========================================================================
    # Architecture Parameters (General)
    # =========================================================================
    model_type: str = "stacked_gru"  # Model architecture type
    filters: int = 64
    kernel_size: tuple = (5, 1)
    hidden_units: int = 128
    dropout_rate: float = 0.2
    activation: str = "relu"
    
    # =========================================================================
    # GRU-Specific Architecture Parameters
    # =========================================================================
    gru_units: tuple = (128, 64)  # Stacked GRU layer sizes
    recurrent_dropout: float = 0.1  # Dropout between recurrent steps
    
    # =========================================================================
    # Multi-Output Task Configuration
    # =========================================================================
    n_routes: int = 3  # Number of route classes (A, C, E)
    regression_loss: str = "huber"  # Loss for headway prediction
    classification_loss: str = "sparse_categorical_crossentropy"  # Loss for route prediction
    huber_delta: float = 1.0  # Delta parameter for Huber loss
    loss_weights: dict = field(default_factory=lambda: {"headway": 1.0, "route": 0.5})  # Output loss weights
    
    # =========================================================================
    # Experiment Tracking Configuration
    # =========================================================================
    experiment_name: str = "headway-prediction-experiments"
    use_vertex_experiments: bool = True
    vertex_location: str = "us-east1"
    track_histograms: bool = True
    histogram_freq: int = 5
    track_profiling: bool = False
    profile_batch_range: tuple = (10, 20)
    
    # =========================================================================
    # Data Configuration
    # =========================================================================
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    
    # =========================================================================
    # Scaling/Normalization
    # =========================================================================
    scaling_method: str = "minmax"  # "minmax", "standard", "robust", "none"
    target_range: tuple = (0.0, 1.0)
    
    # =========================================================================
    # Data Paths
    # =========================================================================
    data_dir: str = "data"
    data_gcs_path: Optional[str] = None
    
    # =========================================================================
    # Output Paths
    # =========================================================================
    model_output_dir: str = "models"
    checkpoint_dir: str = "checkpoints"
    
    # =========================================================================
    # BigQuery Configuration (ETL)
    # =========================================================================
    bq_project: Optional[str] = None
    bq_dataset: Optional[str] = None
    bq_table: Optional[str] = None
    bq_query: Optional[str] = None
    
    # =========================================================================
    # Custom Parameters
    # =========================================================================
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    # =========================================================================
    # Class Methods
    # =========================================================================
    
    @classmethod
    def from_env(cls) -> "ModelConfig":
        """
        Load configuration from environment variables.
        Falls back to defaults if not set.
        
        Environment variables:
            GCP_PROJECT_ID: BigQuery project ID
            VERTEX_LOCATION: Vertex AI location
            TRACK: Track identifier (default: A1)
            ROUTE_IDS: Comma-separated route IDs (default: A,C,E)
            TRAIN_SPLIT: Train split fraction (default: 0.6)
            VAL_SPLIT: Validation split fraction (default: 0.2)
            TEST_SPLIT: Test split fraction (default: 0.2)
            MODEL_NAME: Model identifier
            MODEL_TYPE: Model architecture type
            GRU_UNITS: Comma-separated GRU layer sizes
            DROPOUT_RATE: Dropout rate
            RECURRENT_DROPOUT: Recurrent dropout rate
            LOOKBACK_STEPS: Number of historical timesteps
            BATCH_SIZE: Training batch size
            EPOCHS: Number of training epochs
            LEARNING_RATE: Learning rate
            OPTIMIZER: Optimizer name
            REGRESSION_LOSS: Loss function for regression
            CLASSIFICATION_LOSS: Loss function for classification
            HUBER_DELTA: Huber loss delta parameter
            LOSS_WEIGHTS_HEADWAY: Loss weight for headway output
            LOSS_WEIGHTS_ROUTE: Loss weight for route output
            EXPERIMENT_NAME: Vertex AI experiment name
            USE_VERTEX_EXPERIMENTS: Enable Vertex AI experiments
            TRACK_HISTOGRAMS: Enable histogram logging
            HISTOGRAM_FREQ: Histogram logging frequency
            TRACK_PROFILING: Enable profiling
            PROFILE_BATCH_START: Profiling start batch
            PROFILE_BATCH_END: Profiling end batch
            
        Returns:
            ModelConfig instance with environment overrides
        """
        config = cls()
        
        # GCP Configuration
        if os.getenv("GCP_PROJECT_ID"):
            config.bq_project = os.getenv("GCP_PROJECT_ID")
        
        if os.getenv("VERTEX_LOCATION"):
            config.vertex_location = os.getenv("VERTEX_LOCATION")
        
        # Data Configuration
        if os.getenv("TRACK"):
            config.track = os.getenv("TRACK")
        
        if os.getenv("ROUTE_IDS"):
            route_ids_str = os.getenv("ROUTE_IDS")
            config.route_ids = tuple(r.strip() for r in route_ids_str.split(","))
            config.n_routes = len(config.route_ids)
        
        if os.getenv("TRAIN_SPLIT"):
            config.train_split = float(os.getenv("TRAIN_SPLIT"))
        
        if os.getenv("VAL_SPLIT"):
            config.val_split = float(os.getenv("VAL_SPLIT"))
        
        if os.getenv("TEST_SPLIT"):
            config.test_split = float(os.getenv("TEST_SPLIT"))
        
        # Model Architecture
        if os.getenv("MODEL_NAME"):
            config.model_name = os.getenv("MODEL_NAME")
        
        if os.getenv("MODEL_TYPE"):
            config.model_type = os.getenv("MODEL_TYPE")
        
        if os.getenv("GRU_UNITS"):
            gru_units_str = os.getenv("GRU_UNITS")
            config.gru_units = tuple(int(u.strip()) for u in gru_units_str.split(","))
        
        if os.getenv("DROPOUT_RATE"):
            config.dropout_rate = float(os.getenv("DROPOUT_RATE"))
        
        if os.getenv("RECURRENT_DROPOUT"):
            config.recurrent_dropout = float(os.getenv("RECURRENT_DROPOUT"))
        
        # Training Parameters
        if os.getenv("LOOKBACK_STEPS"):
            config.lookback_steps = int(os.getenv("LOOKBACK_STEPS"))
        
        if os.getenv("BATCH_SIZE"):
            config.batch_size = int(os.getenv("BATCH_SIZE"))
        
        if os.getenv("EPOCHS"):
            config.epochs = int(os.getenv("EPOCHS"))
        
        if os.getenv("LEARNING_RATE"):
            config.learning_rate = float(os.getenv("LEARNING_RATE"))
        
        if os.getenv("OPTIMIZER"):
            config.optimizer = os.getenv("OPTIMIZER")
        
        # Loss Configuration
        if os.getenv("REGRESSION_LOSS"):
            config.regression_loss = os.getenv("REGRESSION_LOSS")
        
        if os.getenv("CLASSIFICATION_LOSS"):
            config.classification_loss = os.getenv("CLASSIFICATION_LOSS")
        
        if os.getenv("HUBER_DELTA"):
            config.huber_delta = float(os.getenv("HUBER_DELTA"))
        
        if os.getenv("LOSS_WEIGHTS_HEADWAY") or os.getenv("LOSS_WEIGHTS_ROUTE"):
            config.loss_weights = {
                "headway": float(os.getenv("LOSS_WEIGHTS_HEADWAY", "1.0")),
                "route": float(os.getenv("LOSS_WEIGHTS_ROUTE", "0.5"))
            }
        
        # Experiment Tracking
        if os.getenv("EXPERIMENT_NAME"):
            config.experiment_name = os.getenv("EXPERIMENT_NAME")
        
        if os.getenv("USE_VERTEX_EXPERIMENTS"):
            config.use_vertex_experiments = os.getenv("USE_VERTEX_EXPERIMENTS").lower() == "true"
        
        if os.getenv("TRACK_HISTOGRAMS"):
            config.track_histograms = os.getenv("TRACK_HISTOGRAMS").lower() == "true"
        
        if os.getenv("HISTOGRAM_FREQ"):
            config.histogram_freq = int(os.getenv("HISTOGRAM_FREQ"))
        
        if os.getenv("TRACK_PROFILING"):
            config.track_profiling = os.getenv("TRACK_PROFILING").lower() == "true"
        
        if os.getenv("PROFILE_BATCH_START") and os.getenv("PROFILE_BATCH_END"):
            config.profile_batch_range = (
                int(os.getenv("PROFILE_BATCH_START")),
                int(os.getenv("PROFILE_BATCH_END"))
            )
        
        return config
    
    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            ModelConfig instance
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ModelConfig instance
        """
        # Handle tuple conversions
        config_dict = config_dict.copy()
        
        if 'kernel_size' in config_dict and isinstance(config_dict['kernel_size'], list):
            config_dict['kernel_size'] = tuple(config_dict['kernel_size'])
        
        if 'target_range' in config_dict and isinstance(config_dict['target_range'], list):
            config_dict['target_range'] = tuple(config_dict['target_range'])
        
        if 'route_ids' in config_dict and isinstance(config_dict['route_ids'], list):
            config_dict['route_ids'] = tuple(config_dict['route_ids'])
        
        # Filter valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation
        """
        config_dict = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            # Convert tuples to lists for JSON serialization
            if isinstance(value, tuple):
                value = list(value)
            config_dict[field_name] = value
        return config_dict
    
    def save_yaml(self, path: str):
        """
        Save configuration to YAML file.
        
        Args:
            path: Output path for YAML file
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: '{key}' is not a valid ModelConfig field")
    
    # =========================================================================
    # Helper Properties
    # =========================================================================
    
    @property
    def total_window_size(self) -> int:
        """Total window size (lookback + forecast)."""
        return self.lookback_steps + self.forecast_steps
    
    @property
    def input_shape(self) -> tuple:
        """Input shape for the model (to be overridden by specific implementations)."""
        raise NotImplementedError("Override this property in task-specific config")
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check splits sum to 1.0
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        
        # Check positive values
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        # Check ranges
        if not (0 <= self.dropout_rate < 1):
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ModelConfig(model_name='{self.model_name}', task_type='{self.task_type}')"
