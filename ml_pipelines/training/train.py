"""Model training for headway prediction."""

import numpy as np
import pandas as pd
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from config.model_config import ModelConfig


class Trainer:
    """Handles model training with time series data."""
    
    def __init__(self, config: "ModelConfig"):
        """
        Initialize trainer.
        
        Args:
            config: ModelConfig instance with training parameters
        """
        self.config = config
        self.input_x = None
        self.input_t = None
        self.input_r = None
    
    def load_data(self, data_path: str) -> None:
        """
        Load preprocessed data and create input arrays.
        
        Args:
            data_path: Path to X.csv (preprocessed features)
        """
        # Load preprocessed data
        data = pd.read_csv(data_path)
        
        # Create input arrays
        self.input_x = data.values  # All features (51751, 8)
        self.input_t = data['log_headway'].values  # Target headway (51751,)
        self.input_r = data[['route_A', 'route_C', 'route_E']].values  # Target route (51751, 3)
        
        print(f"✓ Loaded data:")
        print(f"  input_x: {self.input_x.shape}")
        print(f"  input_t: {self.input_t.shape}")
        print(f"  input_r: {self.input_r.shape}")
    
    def create_datasets(self):
        """Create train/val/test datasets with sliding windows."""
        # TODO: Calculate train/val/test indices
        # TODO: Use keras.utils.timeseries_dataset_from_array
        # TODO: Create datasets with sequence_length=20
        pass
    
    def train(self):
        """Train the model."""
        # TODO: Build model
        # TODO: Compile model
        # TODO: Fit model with datasets
        pass
