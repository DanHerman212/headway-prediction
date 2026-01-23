"""
TensorFlow Dataset Generator

Handles creation of tf.data.Dataset pipelines for efficient training.
"""

from typing import Optional, Tuple, Callable
import numpy as np
import tensorflow as tf


class TFDatasetGenerator:
    """
    Creates optimized tf.data.Dataset pipelines for training.
    
    Features:
    - Efficient data loading with prefetching
    - Batching and shuffling
    - Data augmentation support
    - Time series windowing
    - Multi-input/output support
    
    Example:
        generator = TFDatasetGenerator(
            batch_size=32,
            shuffle_buffer=1000,
            prefetch_buffer=tf.data.AUTOTUNE
        )
        
        # Create dataset from numpy arrays
        train_ds = generator.from_numpy(X_train, y_train, shuffle=True)
        val_ds = generator.from_numpy(X_val, y_val, shuffle=False)
        
        # Create time series windows
        ts_ds = generator.create_time_series_windows(
            data=time_series_data,
            lookback=30,
            forecast=15
        )
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        shuffle_buffer: int = 1000,
        prefetch_buffer: int = tf.data.AUTOTUNE,
        num_parallel_calls: int = tf.data.AUTOTUNE
    ):
        """
        Initialize dataset generator.
        
        Args:
            batch_size: Batch size for training
            shuffle_buffer: Buffer size for shuffling
            prefetch_buffer: Prefetch buffer size (use tf.data.AUTOTUNE for auto)
            num_parallel_calls: Parallel calls for map operations
        """
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.num_parallel_calls = num_parallel_calls
    
    def from_numpy(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        shuffle: bool = False,
        augmentation_fn: Optional[Callable] = None
    ) -> tf.data.Dataset:
        """
        Create tf.data.Dataset from numpy arrays.
        
        Args:
            X: Input features
            y: Target labels (optional)
            shuffle: Whether to shuffle data
            augmentation_fn: Optional data augmentation function
            
        Returns:
            Configured tf.data.Dataset
        """
        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(X)
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
        
        # Apply augmentation if provided
        if augmentation_fn is not None:
            dataset = dataset.map(
                augmentation_fn,
                num_parallel_calls=self.num_parallel_calls
            )
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer)
        
        return dataset
    
    def create_time_series_windows(
        self,
        data: np.ndarray,
        lookback: int,
        forecast: int,
        stride: int = 1,
        shuffle: bool = False,
        batch_size: Optional[int] = None
    ) -> tf.data.Dataset:
        """
        Create sliding windows for time series prediction.
        
        Args:
            data: Time series data (samples, features)
            lookback: Number of historical timesteps
            forecast: Number of future timesteps to predict
            stride: Stride for window creation
            shuffle: Whether to shuffle windows
            batch_size: Batch size (uses instance default if None)
            
        Returns:
            tf.data.Dataset with (input_window, target_window) tuples
        """
        batch_size = batch_size or self.batch_size
        total_window = lookback + forecast
        
        # Create dataset from sequences
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=total_window,
            sequence_stride=stride,
            shuffle=shuffle,
            batch_size=batch_size
        )
        
        # Split into input and target windows
        def split_window(window):
            inputs = window[:, :lookback]
            targets = window[:, lookback:]
            return inputs, targets
        
        dataset = dataset.map(split_window)
        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer)
        
        return dataset
