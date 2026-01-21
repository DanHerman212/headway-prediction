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
    - Dataset with size not divisible by BATCH_SIZE
    - Both timeseries_dataset_from_array AND target dataset must have drop_remainder=True
    - Tests all three splits (train, val, test) to catch validation dataset bugs
    """
    BATCH_SIZE = 64
    LOOKBACK = 20
    N_SAMPLES = 31200  # Simulates real data size (creates ~487 batches + partial)
    
    X = np.random.randn(N_SAMPLES, 8).astype(np.float32)
    
    # Calculate splits
    train_end = int(N_SAMPLES * 0.7)
    val_end = int(N_SAMPLES * 0.85)
    test_end = N_SAMPLES
    
    # Create datasets using the actual function
    train_ds, val_ds, test_ds = create_timeseries_datasets(
        X=X,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end
    )
    
    # Test ALL three datasets (train, val, test) - bug was in validation dataset
    for ds_name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
        batch_sizes_input = []
        batch_sizes_route = []
        batch_sizes_headway = []
        
        for i, (inputs, targets) in enumerate(ds):
            batch_sizes_input.append(inputs.shape[0])
            batch_sizes_route.append(targets['route_output'].shape[0])
            batch_sizes_headway.append(targets['headway_output'].shape[0])
            
            # All batches must match
            assert inputs.shape[0] == targets['route_output'].shape[0], \
                f"{ds_name} batch {i}: input={inputs.shape[0]} != route={targets['route_output'].shape[0]}"
            assert inputs.shape[0] == targets['headway_output'].shape[0], \
                f"{ds_name} batch {i}: input={inputs.shape[0]} != headway={targets['headway_output'].shape[0]}"
            
            # All batches must be BATCH_SIZE (no partial batches)
            assert inputs.shape[0] == BATCH_SIZE, \
                f"{ds_name} batch {i}: Expected {BATCH_SIZE}, got {inputs.shape[0]} (drop_remainder not working)"
        
        # Verify all batches are exactly BATCH_SIZE (drop_remainder working)
        assert len(set(batch_sizes_input)) == 1, f"{ds_name}: Variable input batch sizes detected"
        assert len(set(batch_sizes_route)) == 1, f"{ds_name}: Variable route batch sizes detected"
        assert len(set(batch_sizes_headway)) == 1, f"{ds_name}: Variable headway batch sizes detected"
        
        print(f"✓ {ds_name}: {len(batch_sizes_input)} batches, all size {BATCH_SIZE}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
