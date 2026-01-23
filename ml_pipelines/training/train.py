"""Model training for headway prediction."""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
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
    
    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create train/val/test datasets using index slicing.
        Replaces timeseries_dataset_from_array to avoid TF Graph crashes.
        
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        if self.input_x is None:
            raise ValueError("Must call load_data() before create_datasets()")

        # Prepare data: cast to float32 and convert to TF constants
        input_x_tf = tf.constant(self.input_x.astype(np.float32))
        
        # Prepare targets
        input_t_reshaped = self.input_t.reshape(-1, 1)
        targets_combined = np.column_stack([input_t_reshaped, self.input_r]).astype(np.float32)
        targets_tf = tf.constant(targets_combined)

        sequence_length = self.config.lookback_steps
        batch_size = self.config.batch_size
        n = len(self.input_x)

        # Define slicing logic
        @tf.function
        def get_sequence(index):
            """Get input window and target for given index."""
            # Input window: from index to index + sequence_length
            x_window = input_x_tf[index : index + sequence_length]
            
            # Target: at index + sequence_length (predict next step after window)
            y_val = targets_tf[index + sequence_length]
            
            # Split targets: (headway, route)
            return x_window, (y_val[0:1], y_val[1:4])

        # Helper to build dataset from indices
        def build_from_indices(start_idx, end_idx, is_training):
            """Build dataset from index range."""
            # Calculate valid range: ensure index + sequence_length <= n
            max_possible_idx = n - sequence_length
            
            # Cap the end_idx at max possible
            actual_end = max_possible_idx if end_idx is None else min(end_idx, max_possible_idx)
            
            if start_idx >= actual_end:
                raise ValueError(f"Invalid split: start ({start_idx}) >= end ({actual_end})")

            # Create dataset of indices (lightweight)
            ds = tf.data.Dataset.range(start_idx, actual_end)
            
            if is_training:
                ds = ds.shuffle(buffer_size=10000)

            # Map indices to data slices
            ds = ds.map(get_sequence, num_parallel_calls=tf.data.AUTOTUNE)
            
            # Batch and prefetch
            ds = ds.batch(batch_size, drop_remainder=is_training)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            
            return ds

        # Calculate splits
        train_end = int(n * self.config.train_split)
        val_end = int(n * (self.config.train_split + self.config.val_split))

        print(f"✓ Creating datasets (Index-Based Manual Slicing)...")
        
        # Create datasets
        train_dataset = build_from_indices(0, train_end, is_training=True)
        val_dataset = build_from_indices(train_end, val_end, is_training=False)
        test_dataset = build_from_indices(val_end, None, is_training=False)

        print(f"  Train range: 0 -> {train_end} (target range: {sequence_length} -> {min(train_end + sequence_length, n)})")
        print(f"  Val range:   {train_end} -> {val_end} (target range: {train_end + sequence_length} -> {min(val_end + sequence_length, n)})")
        print(f"  Test range:  {val_end} -> {n - sequence_length} (target range: {val_end + sequence_length} -> {n})")

        return train_dataset, val_dataset, test_dataset
    
    def train(self):
        """Train the model."""
        # TODO: Build model
        # TODO: Compile model
        # TODO: Fit model with datasets
        pass
