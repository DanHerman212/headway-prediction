"""
Configuration schema for experiment tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import yaml


@dataclass
class TrackerConfig:
    """
    Configuration for TensorBoard experiment tracking.
    
    Attributes:
        experiment_name: Name of the experiment (e.g., "headway-regularization")
        run_name: Unique identifier for this run (e.g., "exp01-baseline-20260107")
        log_dir: TensorBoard log directory (local path or gs:// URL)
        
        scalars: Enable scalar metric logging
        histograms: Enable weight/gradient histogram logging
        histogram_freq: How often to log histograms (every N epochs)
        graphs: Enable model graph logging
        hparams: Enable hyperparameter logging
        profiling: Enable GPU/CPU profiling
        profile_batch_range: Batch range to profile (start, end)
        
        hparams_dict: Dictionary of hyperparameters to log
    """
    
    # Experiment identity
    experiment_name: str
    run_name: str
    log_dir: str
    
    # Tracking options
    scalars: bool = True
    histograms: bool = True
    histogram_freq: int = 1
    graphs: bool = True
    hparams: bool = True
    profiling: bool = False
    profile_batch_range: Tuple[int, int] = (10, 20)
    
    # Hyperparameters
    hparams_dict: Dict[str, Any] = field(default_factory=dict)
    
    # Optional metadata
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrackerConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            TrackerConfig instance
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Handle tuple conversion for profile_batch_range
        if 'profile_batch_range' in data and isinstance(data['profile_batch_range'], list):
            data['profile_batch_range'] = tuple(data['profile_batch_range'])
        
        return cls(**data)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrackerConfig":
        """
        Load configuration from a dictionary.
        
        Args:
            d: Configuration dictionary
            
        Returns:
            TrackerConfig instance
        """
        # Handle tuple conversion
        if 'profile_batch_range' in d and isinstance(d['profile_batch_range'], list):
            d = d.copy()
            d['profile_batch_range'] = tuple(d['profile_batch_range'])
        
        return cls(**d)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'experiment_name': self.experiment_name,
            'run_name': self.run_name,
            'log_dir': self.log_dir,
            'scalars': self.scalars,
            'histograms': self.histograms,
            'histogram_freq': self.histogram_freq,
            'graphs': self.graphs,
            'hparams': self.hparams,
            'profiling': self.profiling,
            'profile_batch_range': list(self.profile_batch_range),
            'hparams_dict': self.hparams_dict,
            'description': self.description,
            'tags': self.tags,
        }
    
    def to_yaml(self, path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            path: Path to save YAML file
        """
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
