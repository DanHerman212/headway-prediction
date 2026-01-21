"""Utility functions for artifact handling and validation.

These utilities separate common logic from component code and make debugging easier.
"""
from pathlib import Path
import numpy as np
import logging
from typing import Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_to_artifact(data: np.ndarray, artifact_path: Union[str, Path], filename: str = "data.npy") -> Path:
    """Save numpy array to Kubeflow artifact directory.
    
    Args:
        data: Numpy array to save
        artifact_path: Path to KFP artifact (directory)
        filename: Name of file to create inside directory
        
    Returns:
        Path to saved file
        
    Note:
        KFP artifacts are DIRECTORIES, not files. We must save inside them.
    """
    artifact_dir = Path(artifact_path)
    
    # Validate artifact directory
    if not artifact_dir.exists():
        logger.error(f"Artifact directory does not exist: {artifact_dir}")
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")
    
    if not artifact_dir.is_dir():
        logger.error(f"Artifact path is not a directory: {artifact_dir}")
        raise NotADirectoryError(f"Artifact path must be a directory: {artifact_dir}")
    
    # Save file inside directory
    output_file = artifact_dir / filename
    logger.info(f"Saving artifact to: {output_file}")
    logger.info(f"  Data shape: {data.shape}")
    logger.info(f"  Data dtype: {data.dtype}")
    
    np.save(output_file, data)
    
    # Verify save was successful
    if not output_file.exists():
        logger.error(f"Failed to save artifact: {output_file}")
        raise IOError(f"Failed to save artifact to {output_file}")
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved successfully: {file_size_mb:.2f} MB")
    
    return output_file


def load_from_artifact(artifact_path: Union[str, Path], filename: str = "data.npy") -> np.ndarray:
    """Load numpy array from Kubeflow artifact directory.
    
    Args:
        artifact_path: Path to KFP artifact (directory)
        filename: Name of file to load from directory
        
    Returns:
        Loaded numpy array
    """
    artifact_dir = Path(artifact_path)
    
    # Validate artifact directory
    if not artifact_dir.exists():
        logger.error(f"Artifact directory does not exist: {artifact_dir}")
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")
    
    # Load file from directory
    input_file = artifact_dir / filename
    logger.info(f"Loading artifact from: {input_file}")
    
    if not input_file.exists():
        logger.error(f"Artifact file does not exist: {input_file}")
        raise FileNotFoundError(f"Artifact file not found: {input_file}")
    
    data = np.load(input_file)
    
    logger.info(f"  Loaded shape: {data.shape}")
    logger.info(f"  Loaded dtype: {data.dtype}")
    
    # Validate data quality
    if np.isnan(data).any():
        logger.warning("  WARNING: Data contains NaN values")
    if np.isinf(data).any():
        logger.warning("  WARNING: Data contains inf values")
    
    return data


def validate_preprocessed_data(data: np.ndarray) -> Tuple[bool, list]:
    """Validate preprocessed data format and quality.
    
    Args:
        data: Preprocessed numpy array (n_samples, n_features)
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check dimensionality
    if data.ndim != 2:
        errors.append(f"Expected 2D array, got {data.ndim}D")
    
    # Check feature count
    if data.shape[1] != 8:
        errors.append(f"Expected 8 features, got {data.shape[1]}")
    
    # Check dtype
    if data.dtype != np.float32:
        errors.append(f"Expected float32 dtype, got {data.dtype}")
    
    # Check for invalid values
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        errors.append(f"Data contains {nan_count} NaN values")
    
    if np.isinf(data).any():
        inf_count = np.isinf(data).sum()
        errors.append(f"Data contains {inf_count} inf values")
    
    # Check route one-hot encoding (features 1-3)
    if data.shape[1] >= 4:
        route_one_hot = data[:, 1:4]
        row_sums = route_one_hot.sum(axis=1)
        invalid_rows = np.abs(row_sums - 1.0) > 0.01
        if invalid_rows.any():
            errors.append(f"Route one-hot encoding invalid for {invalid_rows.sum()} rows")
    
    # Check circular features are in valid range [-1, 1]
    if data.shape[1] >= 8:
        for i, name in [(4, 'hour_sin'), (5, 'hour_cos'), (6, 'dow_sin'), (7, 'dow_cos')]:
            if data[:, i].min() < -1.1 or data[:, i].max() > 1.1:
                errors.append(f"{name} out of range: [{data[:, i].min():.2f}, {data[:, i].max():.2f}]")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info("✓ Data validation passed")
    else:
        logger.error(f"✗ Data validation failed with {len(errors)} errors:")
        for error in errors:
            logger.error(f"  - {error}")
    
    return is_valid, errors


def log_dataset_info(dataset, name: str):
    """Log information about a tf.data.Dataset for debugging.
    
    Args:
        dataset: TensorFlow dataset
        name: Name of the dataset (e.g., "train", "val", "test")
    """
    logger.info(f"\n{name} dataset:")
    
    try:
        # Get first batch to inspect structure
        for inputs, targets in dataset.take(1):
            logger.info(f"  Input shape: {inputs.shape}")
            logger.info(f"  Input dtype: {inputs.dtype}")
            
            if isinstance(targets, dict):
                for key, value in targets.items():
                    logger.info(f"  Target '{key}' shape: {value.shape}")
            else:
                logger.info(f"  Target shape: {targets.shape}")
        
        # Count batches
        batch_count = sum(1 for _ in dataset)
        logger.info(f"  Total batches: {batch_count}")
        
    except Exception as e:
        logger.error(f"  Error inspecting dataset: {e}")


if __name__ == "__main__":
    # Test utilities
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test save/load
        artifact_dir = tmpdir / "test_artifact"
        artifact_dir.mkdir()
        
        test_data = np.random.randn(100, 8).astype(np.float32)
        saved_path = save_to_artifact(test_data, artifact_dir, "test.npy")
        loaded_data = load_from_artifact(artifact_dir, "test.npy")
        
        assert np.array_equal(test_data, loaded_data)
        print("✓ Save/load test passed")
        
        # Test validation
        is_valid, errors = validate_preprocessed_data(test_data)
        if not is_valid:
            print(f"Validation errors: {errors}")
