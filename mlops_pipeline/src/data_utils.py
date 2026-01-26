
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional
from src.config import config

class MAESeconds(keras.metrics.Metric):
    """
    Mean Absolute Error in seconds (converted from log-space).
    Moved here to be shared between Train and Eval.
    """
    def __init__(self, name='mae_seconds', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_seconds = tf.exp(y_true)
        y_pred_seconds = tf.exp(y_pred)
        errors = tf.abs(y_true_seconds - y_pred_seconds)
        if sample_weight is not None:
            errors = errors * sample_weight
        self.total_error.assign_add(tf.reduce_sum(errors))
        if sample_weight is not None:
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(errors), dtype=tf.float32))
    
    def result(self):
        return self.total_error / self.count
    
    def reset_state(self):
        self.total_error.assign(0.0)
        self.count.assign(0.0)
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def inverse_transform_headway(log_headway: np.ndarray) -> np.ndarray:
    """
    Reverses the log1p transformation: exp(x) - 1.
    """
    return np.expm1(log_headway)

def prepare_tensors(df: pd.DataFrame) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Converts DataFrame into TF Constants for Inputs (X) and Targets (Y).
    
    Returns:
        input_x_tf: Tensor of shape (N, features)
        targets_tf: Tensor of shape (N, 1 + num_routes) -> [log_headway, route_0, route_1...]
    """
    # Input Features
    # Note: We assume DataFrame columns are already correct from preprocessing
    # The order of columns in X must match what the model expects.
    # In preprocess.py we saved: log_headway, route_*, hour_*, day_*, scheduled_headway
    # We explicitly drop 'scheduled_headway' as it is only for evaluation metadata
    feature_df = df.drop(columns=['scheduled_headway'], errors='ignore')
    input_x = feature_df.values.astype(np.float32)

    # Targets
    # Headway is 'log_headway'
    input_t = df['log_headway'].values
    
    # Route columns
    route_cols = [c for c in df.columns if c.startswith('route_')]
    input_r = df[route_cols].values
    
    # Targets Combined: [headway, route_0, route_1, ...]
    input_t_reshaped = input_t.reshape(-1, 1)
    targets_combined = np.column_stack([input_t_reshaped, input_r]).astype(np.float32)
    
    input_x_tf = tf.constant(input_x)
    targets_tf = tf.constant(targets_combined)
    
    return input_x_tf, targets_tf

def create_windowed_dataset(
    df: pd.DataFrame, 
    batch_size: int, 
    lookback_steps: int, 
    start_index: int = 0, 
    end_index: Optional[int] = None, 
    shuffle: bool = False
) -> tf.data.Dataset:
    """
    Creates a sliding window dataset from the given DataFrame.
    
    Args:
        df: The preprocessed dataframe.
        batch_size: Batch size.
        lookback_steps: Sequence length (N).
        start_index: Row index to start windowing from.
        end_index: Row index to end windowing at (exclusive). If None, goes to end.
        shuffle: Whether to shuffle the windows.
        
    Returns:
        tf.data.Dataset yielding (x_window, (y_headway, y_route))
    """
    input_x_tf, targets_tf = prepare_tensors(df)
    
    n = len(df)
    
    # Identify number of route columns for splitting targets later
    route_cols = [c for c in df.columns if c.startswith('route_')]
    num_route_cols = len(route_cols)

    # Max possible start index for a window
    # Valid window at index i: X[i : i+N] -> Target Y[i+N]
    # Max i is such that i+N < n (so we have a target at n-1 is max? No, target at i+N)
    # If len=100, lookback=10. Last window starts at 89?
    # X[89:99] (len 10), Target Y[99]. Correct.
    # So max_idx = n - lookback_steps
    max_possible_idx = n - lookback_steps
    
    actual_end = min(end_index, max_possible_idx) if end_index is not None else max_possible_idx
    
    if start_index >= actual_end:
        # Case where split is too small for windowing or explicit range is invalid
        # Return empty dataset or raise? Returning empty is safer for edge cases.
        # But if it's strictly invalid (start > end), empty is fine.
        print(f"Warning: Dataset range {start_index}:{actual_end} is invalid or empty given lookback {lookback_steps}.")
        return tf.data.Dataset.range(0).take(0)

    @tf.function
    def get_sequence(index):
        # Window: indices [index, index + N)  -> Shape (N, features)
        x_window = input_x_tf[index : index + lookback_steps]
        
        # Target: index + N
        y_val = targets_tf[index + lookback_steps]
        
        # Split y_val into (headway, route)
        # headway is y_val[0], route is y_val[1:]
        # Return tuple matching model outputs (headway_out, route_out)
        return x_window, (y_val[0:1], y_val[1:1+num_route_cols])

    ds = tf.data.Dataset.range(start_index, actual_end)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
        
    ds = ds.map(get_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=shuffle)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds
