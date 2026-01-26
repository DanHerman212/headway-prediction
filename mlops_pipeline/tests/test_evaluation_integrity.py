
import os
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import tensorflow as tf
from src.eval import evaluate_model
from src.data_utils import inverse_transform_headway
from src.config import config

# Constants for the test
TEST_DIR = "tests/artifacts/integrity"
MODEL_PATH = f"{TEST_DIR}/mock_model.keras"
DATA_PATH = f"{TEST_DIR}/test_set.csv"
METRICS_PATH = f"{TEST_DIR}/metrics.json"

@pytest.fixture
def clean_artifacts():
    if os.path.exists(TEST_DIR):
        import shutil
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    # Cleanup optional

def test_evaluation_integrity(clean_artifacts):
    """
    Verifies the mathematical integrity of the evaluation pipeline.
    
    Scenario:
    - 2 Samples
    - True Headways: [2.0, 4.0] minutes
    - Scheduled:     [5.0, 5.0] minutes
    - Predicted:     [3.0, 3.0] minutes (Model predicts constant 3.0)
    
    Calculations (Seconds):
    - True:     [120.0, 240.0]
    - Pred:     [180.0, 180.0]
    - Sched:    [300.0, 300.0]
    
    Errors:
    - Model Error:     |120-180| = 60, |240-180| = 60. AVG = 60.0 seconds.
    - Baseline Error:  |120-300| = 180, |240-300| = 60. AVG = 120.0 seconds.
    - Reduction:       1 - (60/120) = 0.5 (50%)
    """
    
    # 1. Create Test Data
    # config.lookback_steps needs to be handled.
    # Dataset generator expects lookback_steps + N rows.
    lookback = config.lookback_steps
    n_samples = 2
    total_rows = lookback + n_samples
    
    # Create timestamps/features
    rows = []
    
    # We only care about the last 2 rows for targets (indices `lookback` and `lookback+1`)
    # But we need input data for them relative to the window.
    # The eval script slices `y_true = input[lookback:]`
    
    true_minutes = [2.0, 4.0]
    sched_minutes = [5.0, 5.0]
    
    # Helper to reverse log1p for file creation
    def to_log(m): return np.log1p(m)
    
    # Fill setup rows (0 to lookback-1) with dummy data
    for i in range(lookback):
        rows.append({
            'log_headway': to_log(2.0),
            'scheduled_headway': 5.0,
            'route_A': 1, 'route_C': 0, 'route_E': 0,
            'hour_sin': 0, 'hour_cos': 1, 'day_sin': 0, 'day_cos': 1
        })
        
    # Fill target rows (lookback to lookback+1)
    for i in range(n_samples):
        rows.append({
            'log_headway': to_log(true_minutes[i]),
            'scheduled_headway': sched_minutes[i],
            'route_A': 1, 'route_C': 0, 'route_E': 0,
            'hour_sin': 0, 'hour_cos': 1, 'day_sin': 0, 'day_cos': 1
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    
    # 2. Mock Model
    # We need a compiled Keras model signature, but we want to control output.
    # Easiest way is to patch `predict` method of the returned object from load_model.
    
    mock_model = MagicMock()
    
    # Predicted Headway (Log Space)
    # We want predictions to be 3.0 minutes -> log1p(3.0)
    pred_val = to_log(3.0)
    
    # Output shape: List of [Headway, Route] (Multi-output)
    # Shape must match (n_samples, 1) and (n_samples, 3)
    pred_headway = np.full((n_samples, 1), pred_val)
    pred_route = np.zeros((n_samples, 3)) 
    pred_route[:, 0] = 1 # Predict Route A
    
    mock_model.predict.return_value = [pred_headway, pred_route]
    
    # Patch load_model to return our mock
    # We patch 'src.eval.keras.models.load_model' because eval.py imports keras from tensorflow
    # Actually, simpler to patch the method where it is CALLED if possible, or the exact object.
    # Given 'from tensorflow import keras', we patch the function on that object.
    with patch('src.eval.keras.models.load_model', return_value=mock_model):
        with patch('google.cloud.aiplatform.init'), patch('google.cloud.aiplatform.start_run'):
            # Run Evaluation
            evaluate_model(MODEL_PATH, DATA_PATH, METRICS_PATH)
            
    # 3. Verify Results
    assert os.path.exists(METRICS_PATH)
    
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
        
    print("\n--- Integrity Test Results ---")
    print(json.dumps(metrics, indent=2))
    
    # Assertions
    # MAE Seconds: Expected 60.0
    # Allow small float tolerance
    assert abs(metrics['mae_seconds'] - 60.0) < 0.1, f"MAE Integrity Fail: Expected 60.0, got {metrics['mae_seconds']}"
    
    # Baseline MAE Seconds: Expected 120.0
    assert abs(metrics['baseline_mae_seconds'] - 120.0) < 0.1, f"Baseline MAE Integrity Fail: Expected 120.0, got {metrics['baseline_mae_seconds']}"
    
    # Reduction: Expected 0.5
    assert abs(metrics['mae_reduction'] - 0.5) < 0.001, f"Reduction Integrity Fail: Expected 0.5, got {metrics['mae_reduction']}"
    
    print("\n✅ INTEGRITY CONFIRMED: Calculations are mathematically precise.")
