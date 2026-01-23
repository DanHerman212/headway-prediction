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
    # Architecture Parameters
    # =========================================================================
    filters: int = 64
    kernel_size: tuple = (5, 1)
    hidden_units: int = 128
    dropout_rate: float = 0.2
    activation: str = "relu"
    
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
        if 'kernel_size' in config_dict and isinstance(config_dict['kernel_size'], list):
            config_dict = config_dict.copy()
            config_dict['kernel_size'] = tuple(config_dict['kernel_size'])
        
        if 'target_range' in config_dict and isinstance(config_dict['target_range'], list):
            config_dict = config_dict.copy()
            config_dict['target_range'] = tuple(config_dict['target_range'])
        
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
