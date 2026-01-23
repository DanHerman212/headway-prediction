"""
Tracking Configuration

Configuration schema for Vertex AI Experiments and TensorBoard tracking.
Defines what metrics, histograms, graphs, and profiling data to capture.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import yaml
from datetime import datetime


@dataclass
class TrackingConfig:
    """
    Configuration for experiment tracking with Vertex AI and TensorBoard.
    
    This configuration determines what data is logged during training:
    - Scalars: Loss, metrics, learning rate, custom metrics
    - Histograms: Weight distributions, gradient distributions
    - Graphs: Model architecture visualization
    - HParams: Hyperparameter comparison
    - Profiling: GPU/CPU performance analysis
    - Images: Custom visualizations (predictions, attention maps, etc.)
    
    Attributes:
        # Experiment Identity
        experiment_name: Name of the Vertex AI Experiment
        run_name: Unique identifier for this run
        description: Human-readable description of the experiment
        tags: Key-value tags for organization
        
        # Log Directory
        log_dir: TensorBoard log directory (local path or gs:// URL)
        
        # Tracking Toggles
        scalars: Enable scalar metric logging
        histograms: Enable weight/gradient histogram logging
        histogram_freq: Log histograms every N epochs
        graphs: Enable model architecture graph logging
        hparams: Enable hyperparameter tracking
        profiling: Enable GPU/CPU profiling
        profile_batch_range: Batch range to profile (start, end)
        images: Enable image logging (for visualizations)
        
        # Vertex AI Configuration
        use_vertex_experiments: Whether to use Vertex AI Experiments
        vertex_project: GCP project ID for Vertex AI
        vertex_location: GCP region for Vertex AI
        tensorboard_resource_name: Full resource name of Vertex AI TensorBoard instance
        
        # Hyperparameters (logged to both TensorBoard and Vertex AI)
        hparams_dict: Dictionary of hyperparameters to track
        
        # Metadata
        created_at: Timestamp of configuration creation
    """
    
    # =========================================================================
    # Experiment Identity
    # =========================================================================
    experiment_name: str
    run_name: str
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    # =========================================================================
    # Log Directory
    # =========================================================================
    log_dir: str = "tensorboard_logs"
    
    # =========================================================================
    # Tracking Toggles
    # =========================================================================
    scalars: bool = True
    histograms: bool = True
    histogram_freq: int = 1
    graphs: bool = True
    hparams: bool = True
    profiling: bool = False
    profile_batch_range: Tuple[int, int] = (10, 20)
    images: bool = False
    
    # =========================================================================
    # Vertex AI Configuration
    # =========================================================================
    use_vertex_experiments: bool = True
    vertex_project: Optional[str] = None
    vertex_location: str = "us-east1"
    tensorboard_resource_name: Optional[str] = None
    
    # For Vertex AI TensorBoard instance creation/management
    tensorboard_display_name: Optional[str] = None
    
    # =========================================================================
    # Hyperparameters
    # =========================================================================
    hparams_dict: Dict[str, Any] = field(default_factory=dict)
    
    # =========================================================================
    # Metadata
    # =========================================================================
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # =========================================================================
    # Class Methods
    # =========================================================================
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrackingConfig":
        """
        Load tracking configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            TrackingConfig instance
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrackingConfig":
        """
        Create tracking configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            TrackingConfig instance
        """
        # Handle tuple conversion for profile_batch_range
        if 'profile_batch_range' in config_dict and isinstance(config_dict['profile_batch_range'], list):
            config_dict = config_dict.copy()
            config_dict['profile_batch_range'] = tuple(config_dict['profile_batch_range'])
        
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
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def create_from_model_config(
        cls,
        model_config: "ModelConfig",
        experiment_name: str,
        run_name: Optional[str] = None,
        **kwargs
    ) -> "TrackingConfig":
        """
        Create TrackingConfig from ModelConfig, extracting relevant hyperparameters.
        
        Args:
            model_config: ModelConfig instance
            experiment_name: Name for the Vertex AI Experiment
            run_name: Unique run identifier (auto-generated if None)
            **kwargs: Additional tracking configuration parameters
            
        Returns:
            TrackingConfig instance with hyperparameters from model_config
        """
        # Auto-generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"{model_config.model_name}-{timestamp}"
        
        # Extract hyperparameters from model config
        hparams_dict = {
            "model_name": model_config.model_name,
            "task_type": model_config.task_type,
            "lookback_steps": model_config.lookback_steps,
            "forecast_steps": model_config.forecast_steps,
            "batch_size": model_config.batch_size,
            "learning_rate": model_config.learning_rate,
            "epochs": model_config.epochs,
            "optimizer": model_config.optimizer,
            "loss": model_config.loss,
            "filters": model_config.filters,
            "dropout_rate": model_config.dropout_rate,
        }
        
        # Add custom params if present
        if model_config.custom_params:
            hparams_dict.update(model_config.custom_params)
        
        # Create log directory based on experiment and run name
        log_dir = kwargs.pop("log_dir", f"tensorboard_logs/{experiment_name}/{run_name}")
        
        return cls(
            experiment_name=experiment_name,
            run_name=run_name,
            log_dir=log_dir,
            hparams_dict=hparams_dict,
            description=model_config.description,
            **kwargs
        )
    
    def get_tensorboard_command(self) -> str:
        """
        Get the command to launch TensorBoard for this configuration.
        
        Returns:
            TensorBoard launch command string
        """
        return f"tensorboard --logdir={self.log_dir}"
    
    def validate(self) -> bool:
        """
        Validate tracking configuration.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        if not self.experiment_name:
            raise ValueError("experiment_name is required")
        
        if not self.run_name:
            raise ValueError("run_name is required")
        
        if not self.log_dir:
            raise ValueError("log_dir is required")
        
        if self.histogram_freq < 1:
            raise ValueError(f"histogram_freq must be >= 1, got {self.histogram_freq}")
        
        if self.profile_batch_range[0] >= self.profile_batch_range[1]:
            raise ValueError(
                f"Invalid profile_batch_range: start must be < end, got {self.profile_batch_range}"
            )
        
        if self.use_vertex_experiments and not self.vertex_project:
            raise ValueError("vertex_project is required when use_vertex_experiments=True")
        
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TrackingConfig(experiment='{self.experiment_name}', run='{self.run_name}')"
