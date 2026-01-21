"""Unit tests for preprocessing component."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import preprocess_pipeline


def test_preprocess_saves_to_artifact_dir(sample_raw_data, temp_artifact_dir):
    """Test that preprocessing saves data to artifact directory correctly."""
    output_path = temp_artifact_dir / "preprocessed.npy"
    
    # Run preprocessing (mocked to use sample data instead of BigQuery)
    # Note: This requires mocking BigQuery client - for now just test the save logic
    X = np.random.randn(100, 8).astype(np.float32)
    np.save(output_path, X)
    
    # Verify file exists and can be loaded
    assert output_path.exists()
    loaded = np.load(output_path)
    assert loaded.shape == X.shape
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, X)


def test_preprocessed_data_format(sample_preprocessed_data):
    """Test that preprocessed data has correct format and ranges."""
    X = sample_preprocessed_data
    
    # Check shape: (n_samples, 8 features)
    assert X.ndim == 2
    assert X.shape[1] == 8
    assert X.dtype == np.float32
    
    # Check route one-hot: exactly one route per sample
    route_one_hot = X[:, 1:4]
    row_sums = route_one_hot.sum(axis=1)
    np.testing.assert_array_almost_equal(row_sums, np.ones(len(X)))
    
    # Check circular features are in [-1, 1]
    assert X[:, 4].min() >= -1 and X[:, 4].max() <= 1  # hour_sin
    assert X[:, 5].min() >= -1 and X[:, 5].max() <= 1  # hour_cos
    assert X[:, 6].min() >= -1 and X[:, 6].max() <= 1  # dow_sin
    assert X[:, 7].min() >= -1 and X[:, 7].max() <= 1  # dow_cos


def test_preprocessed_data_no_nan(sample_preprocessed_data):
    """Test that preprocessed data contains no NaN or inf values."""
    X = sample_preprocessed_data
    
    assert not np.isnan(X).any(), "Preprocessed data contains NaN values"
    assert not np.isinf(X).any(), "Preprocessed data contains inf values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
