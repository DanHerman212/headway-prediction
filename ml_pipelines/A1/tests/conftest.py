"""Pytest configuration and shared fixtures for A1 pipeline tests."""
import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_raw_data():
    """Create sample raw data matching BigQuery schema."""
    np.random.seed(42)
    n_samples = 1000
    
    return {
        'trip_id': [f'trip_{i}' for i in range(n_samples)],
        'route_id': np.random.choice(['A', 'C', 'E'], n_samples),
        'stop_sequence': np.random.randint(1, 30, n_samples),
        'arrival_time': np.arange(n_samples) * 60,  # One minute apart
        'departure_time': np.arange(n_samples) * 60 + 30,
        'stop_id': [f'stop_{i % 10}' for i in range(n_samples)],
        'headway_seconds': np.random.uniform(120, 600, n_samples),
    }


@pytest.fixture
def sample_preprocessed_data():
    """Create sample preprocessed numpy array matching expected format.
    
    Features: [log_headway, route_A, route_C, route_E, hour_sin, hour_cos, dow_sin, dow_cos]
    """
    np.random.seed(42)
    n_samples = 1000
    n_features = 8
    
    # Create realistic data
    data = np.zeros((n_samples, n_features), dtype=np.float32)
    
    # log_headway: between log(120) and log(600)
    data[:, 0] = np.random.uniform(np.log(120), np.log(600), n_samples)
    
    # route one-hot: randomly assign one route per sample
    routes = np.random.choice([0, 1, 2], n_samples)
    for i, route in enumerate(routes):
        data[i, 1 + route] = 1.0
    
    # hour_sin, hour_cos: simulate 24-hour cycle
    hours = np.random.uniform(0, 24, n_samples)
    data[:, 4] = np.sin(2 * np.pi * hours / 24)
    data[:, 5] = np.cos(2 * np.pi * hours / 24)
    
    # dow_sin, dow_cos: simulate 7-day cycle
    days = np.random.uniform(0, 7, n_samples)
    data[:, 6] = np.sin(2 * np.pi * days / 7)
    data[:, 7] = np.cos(2 * np.pi * days / 7)
    
    return data


@pytest.fixture
def temp_artifact_dir(tmp_path):
    """Create temporary directory for artifact testing."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    return artifact_dir


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    class MockConfig:
        BQ_PROJECT = "test-project"
        BQ_DATASET = "test_dataset"
        BQ_TABLE = "test_table"
        LOOKBACK_WINDOW = 20
        FORECAST_HORIZON = 1
        BATCH_SIZE = 32  # Smaller for testing
        EPOCHS = 2  # Fewer for testing
        TRAIN_SPLIT = 0.6
        VAL_SPLIT = 0.2
        EXPERIMENT_NAME = "test-experiment"
        TENSORBOARD_LOG_DIR = "gs://test-bucket/logs"
    
    return MockConfig()
