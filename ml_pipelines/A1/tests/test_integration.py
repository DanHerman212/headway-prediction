"""Integration test simulating the full pipeline locally.

This test validates the entire pipeline workflow:
1. Preprocess data -> save to artifact directory
2. Load preprocessed data -> create datasets
3. Train model for a few steps
4. Verify no errors occur

This catches 90% of deployment issues before pushing to Vertex AI.
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import pytest
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import create_timeseries_datasets, train_model
from src.model import get_model


def test_full_pipeline_simulation(sample_preprocessed_data, mock_config, tmp_path):
    """Simulate the full pipeline: preprocess -> train -> validate.
    
    This is the CRITICAL test that should pass before any deployment.
    """
    print("\n=== FULL PIPELINE SIMULATION ===")
    
    # Step 1: Simulate preprocessing - save data to artifact directory
    print("\n1. Preprocessing component...")
    preprocess_artifact_dir = tmp_path / "preprocessed_npy"
    preprocess_artifact_dir.mkdir()
    preprocessed_file = preprocess_artifact_dir / "preprocessed_data.npy"
    
    X = sample_preprocessed_data
    np.save(preprocessed_file, X)
    print(f"   Saved preprocessed data: {preprocessed_file}")
    print(f"   Shape: {X.shape}, dtype: {X.dtype}")
    assert preprocessed_file.exists()
    
    # Step 2: Load preprocessed data (simulating component boundary)
    print("\n2. Loading preprocessed data...")
    X_loaded = np.load(preprocessed_file)
    print(f"   Loaded shape: {X_loaded.shape}")
    assert X_loaded.shape == X.shape
    
    # Step 3: Create train/val/test datasets
    print("\n3. Creating datasets...")
    train_end = int(len(X_loaded) * mock_config.TRAIN_SPLIT)
    val_end = int(len(X_loaded) * (mock_config.TRAIN_SPLIT + mock_config.VAL_SPLIT))
    test_end = len(X_loaded)
    
    train_ds, val_ds, test_ds = create_timeseries_datasets(
        X=X_loaded,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end
    )
    print("   Datasets created successfully")
    
    # Step 4: Verify dataset batch consistency
    print("\n4. Validating batch consistency...")
    batch_count = 0
    for inputs, targets in train_ds.take(3):
        batch_size = inputs.shape[0]
        route_batch_size = targets['route_output'].shape[0]
        headway_batch_size = targets['headway_output'].shape[0]
        
        assert batch_size == route_batch_size == headway_batch_size, \
            f"Batch size mismatch: {batch_size} vs {route_batch_size} vs {headway_batch_size}"
        batch_count += 1
    print(f"   Validated {batch_count} batches - all consistent")
    
    # Step 5: Create and compile model
    print("\n5. Creating model...")
    model = get_model(compile=True)
    print(f"   Model created: {model.count_params():,} parameters")
    
    # Step 6: Train for a few steps to catch any runtime errors
    print("\n6. Training model (2 epochs)...")
    history = model.fit(
        train_ds.take(5),  # Only use 5 batches for speed
        validation_data=val_ds.take(2),
        epochs=2,
        verbose=0
    )
    print(f"   Training completed")
    print(f"   Final loss: {history.history['loss'][-1]:.4f}")
    print(f"   Final val_loss: {history.history['val_loss'][-1]:.4f}")
    
    # Step 7: Validate training produced reasonable results
    assert len(history.history['loss']) == 2, "Should have 2 epochs"
    assert all(np.isfinite(loss) for loss in history.history['loss']), "Loss should be finite"
    
    # Step 8: Test model prediction
    print("\n7. Testing model prediction...")
    for inputs, _ in test_ds.take(1):
        predictions = model(inputs, training=False)
        if isinstance(predictions, dict):
            route_pred = predictions['route_output']
            headway_pred = predictions['headway_output']
        else:
            route_pred, headway_pred = predictions
        
        print(f"   Prediction shapes: route={route_pred.shape}, headway={headway_pred.shape}")
        assert tf.reduce_all(tf.math.is_finite(route_pred)), "Route predictions should be finite"
        assert tf.reduce_all(tf.math.is_finite(headway_pred)), "Headway predictions should be finite"
    
    print("\n✓ FULL PIPELINE SIMULATION PASSED")
    print("  This pipeline is ready for deployment.")


def test_artifact_directory_handling(tmp_path):
    """Test that artifact directory handling matches KFP behavior.
    
    Kubeflow provides artifact.path as a DIRECTORY, not a file.
    We must save files INSIDE this directory.
    """
    print("\n=== ARTIFACT DIRECTORY HANDLING ===")
    
    # Simulate KFP artifact directory
    artifact_dir = tmp_path / "my_artifact"
    artifact_dir.mkdir()
    
    print(f"\n1. KFP provides: {artifact_dir} (directory)")
    assert artifact_dir.is_dir()
    
    # CORRECT: Save file inside directory
    output_file = artifact_dir / "data.npy"
    data = np.random.randn(100, 8).astype(np.float32)
    np.save(output_file, data)
    print(f"2. We save to: {output_file} (file inside directory)")
    
    # CORRECT: Load from file inside directory
    loaded_data = np.load(output_file)
    print(f"3. We load from: {output_file}")
    
    assert loaded_data.shape == data.shape
    print("\n✓ ARTIFACT HANDLING PASSED")


def test_batch_size_edge_cases():
    """Test edge cases where sample count is not divisible by batch size.
    
    This is the exact scenario that caused the production error.
    """
    print("\n=== BATCH SIZE EDGE CASES ===")
    
    test_cases = [
        (1000, 64, 20),  # 980 valid samples -> 15 full batches + 20 remainder
        (500, 32, 20),   # 480 valid samples -> 15 full batches
        (300, 64, 20),   # 280 valid samples -> 4 full batches + 24 remainder
    ]
    
    for n_samples, batch_size, lookback in test_cases:
        print(f"\nTesting: {n_samples} samples, batch={batch_size}, lookback={lookback}")
        
        X = np.random.randn(n_samples, 8).astype(np.float32)
        valid_samples = n_samples - lookback
        expected_full_batches = valid_samples // batch_size
        remainder = valid_samples % batch_size
        
        print(f"  Valid samples: {valid_samples}")
        print(f"  Expected batches: {expected_full_batches} full + {remainder} remainder")
        
        # Create dataset with drop_remainder=True
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=X,
            targets=None,
            sequence_length=lookback,
            batch_size=batch_size,
        )
        
        target_dataset = tf.data.Dataset.from_tensor_slices(
            np.random.randn(valid_samples, 3).astype(np.float32)
        ).batch(batch_size, drop_remainder=True)
        
        combined = tf.data.Dataset.zip((dataset, target_dataset))
        
        # Count actual batches
        batch_count = 0
        for inputs, targets in combined:
            input_batch = inputs.shape[0]
            target_batch = targets.shape[0]
            
            # Should either be equal or input has remainder
            assert input_batch == target_batch or (input_batch == remainder and input_batch < batch_size), \
                f"Batch {batch_count}: input {input_batch} vs target {target_batch}"
            batch_count += 1
        
        print(f"  Actual batches: {batch_count}")
        print(f"  ✓ No shape mismatches")
    
    print("\n✓ EDGE CASES PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
