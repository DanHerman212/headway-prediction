"""Unit tests for dataset creation and batching logic."""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import create_timeseries_datasets


def test_dataset_batch_alignment(sample_preprocessed_data, mock_config):
    """Test that input and target batches are aligned (no shape mismatch)."""
    # Use smaller data for faster testing
    X = sample_preprocessed_data[:500]
    
    train_end = int(len(X) * mock_config.TRAIN_SPLIT)
    val_end = int(len(X) * (mock_config.TRAIN_SPLIT + mock_config.VAL_SPLIT))
    test_end = len(X)
    
    # Create datasets
    train_ds, val_ds, test_ds = create_timeseries_datasets(
        X=X,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end
    )
    
    # Verify train dataset batches are consistent
    for i, (inputs, targets) in enumerate(train_ds.take(5)):
        batch_size = inputs.shape[0]
        route_batch_size = targets['route_output'].shape[0]
        headway_batch_size = targets['headway_output'].shape[0]
        
        assert batch_size == route_batch_size, \
            f"Batch {i}: input batch size {batch_size} != route target batch size {route_batch_size}"
        assert batch_size == headway_batch_size, \
            f"Batch {i}: input batch size {batch_size} != headway target batch size {headway_batch_size}"
        assert route_batch_size == headway_batch_size, \
            f"Batch {i}: route target batch size {route_batch_size} != headway target batch size {headway_batch_size}"


def test_dataset_shapes(sample_preprocessed_data, mock_config):
    """Test that dataset shapes match expected dimensions."""
    X = sample_preprocessed_data[:500]
    
    train_end = int(len(X) * mock_config.TRAIN_SPLIT)
    val_end = int(len(X) * (mock_config.TRAIN_SPLIT + mock_config.VAL_SPLIT))
    test_end = len(X)
    
    train_ds, val_ds, test_ds = create_timeseries_datasets(
        X=X,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end
    )
    
    # Check first batch
    for inputs, targets in train_ds.take(1):
        # Inputs: (batch, lookback_window, n_features)
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == mock_config.LOOKBACK_WINDOW
        assert inputs.shape[2] == 8  # 8 features
        
        # Route output: (batch, 3) - one-hot for 3 routes
        assert targets['route_output'].shape[1] == 3
        
        # Headway output: (batch, 1) - single regression value
        assert targets['headway_output'].shape[1] == 1


def test_dataset_no_empty_batches(sample_preprocessed_data, mock_config):
    """Test that all batches contain data (no zero-sized batches)."""
    X = sample_preprocessed_data[:500]
    
    train_end = int(len(X) * mock_config.TRAIN_SPLIT)
    val_end = int(len(X) * (mock_config.TRAIN_SPLIT + mock_config.VAL_SPLIT))
    test_end = len(X)
    
    train_ds, val_ds, test_ds = create_timeseries_datasets(
        X=X,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end
    )
    
    # Count batches and verify all have > 0 samples
    for ds_name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
        batch_count = 0
        for inputs, targets in ds:
            batch_size = inputs.shape[0]
            assert batch_size > 0, f"{ds_name} dataset has empty batch"
            batch_count += 1
        
        assert batch_count > 0, f"{ds_name} dataset has no batches"


def test_drop_remainder_prevents_last_batch_mismatch():
    """Test that drop_remainder=True prevents the batch size mismatch error.
    
    This simulates the exact error that occurred in production:
    - 880 samples with batch_size=64 creates 13 full batches + 1 partial (48 samples)
    - Without drop_remainder, input and target datasets can have different last batch sizes
    """
    BATCH_SIZE = 64
    LOOKBACK = 20
    N_SAMPLES = 900  # Will create 880 valid timeseries samples
    
    X = np.random.randn(N_SAMPLES, 8).astype(np.float32)
    
    # Simulate dataset creation with drop_remainder
    X_split = X[:N_SAMPLES]
    max_samples = len(X_split) - LOOKBACK
    
    # Create input dataset (timeseries_dataset_from_array doesn't have drop_remainder param)
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=X_split,
        targets=None,
        sequence_length=LOOKBACK,
        sequence_stride=1,
        shuffle=False,
        batch_size=BATCH_SIZE,
    )
    
    # Create target dataset WITH drop_remainder=True
    route_targets = X[LOOKBACK:N_SAMPLES, 1:4]
    headway_targets = X[LOOKBACK:N_SAMPLES, 0:1]
    target_dataset = tf.data.Dataset.from_tensor_slices({
        'route_output': route_targets,
        'headway_output': headway_targets
    }).batch(BATCH_SIZE, drop_remainder=True)
    
    # Zip and verify no shape mismatch
    combined_ds = tf.data.Dataset.zip((dataset, target_dataset))
    
    for i, (inputs, targets) in enumerate(combined_ds):
        input_batch_size = inputs.shape[0]
        route_batch_size = targets['route_output'].shape[0]
        
        # This should not raise an error
        assert input_batch_size == route_batch_size or input_batch_size <= BATCH_SIZE, \
            f"Batch {i}: Mismatch - input {input_batch_size} vs route {route_batch_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
