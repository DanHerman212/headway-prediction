# -*- coding: utf-8 -*-
"""
Quick validation test for A1 pipeline modules.

Tests each module can import and execute basic operations without errors.
Does NOT require BigQuery access or training data.

Usage:
    cd ml_pipelines/A1
    python test_modules.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

print("="*80)
print("A1 Pipeline Module Validation Test")
print("="*80)
print()

# Test 1: Config
print("Test 1: Config Module")
print("-" * 40)
try:
    from config import config
    print("[OK] Config imported")
    print("  Lookback window: {}".format(config.LOOKBACK_WINDOW))
    print("  Batch size: {}".format(config.BATCH_SIZE))
    print("  GRU units: {} -> {}".format(config.GRU_UNITS_1, config.GRU_UNITS_2))
    print("  BQ Project: {}".format(config.BQ_PROJECT))
    print("  TensorBoard: {}".format(config.TENSORBOARD_LOG_DIR))
    print("[OK] Config test passed\n")
except Exception as e:
    print("[FAIL] Config test failed: {}\n".format(e))
    sys.exit(1)

# Test 2: Model
print("Test 2: Model Module")
print("-" * 40)
try:
    from model import build_stacked_gru_model, compile_model, mae_seconds
    print("[OK] Model functions imported")
    
    # Build model
    model = build_stacked_gru_model()
    print("[OK] Model built successfully")
    
    # Compile model
    model = compile_model(model)
    print("[OK] Model compiled")
    
    # Check architecture
    total_params = model.count_params()
    print("  Total parameters: {:,}".format(total_params))
    print("[OK] Model test passed\n")
except Exception as e:
    print("[FAIL] Model test failed: {}\n".format(e))
    sys.exit(1)

# Test 3: Preprocessing functions (without real data)
print("Test 3: Preprocessing Module")
print("-" * 40)
try:
    from preprocess import (
        log_transform_headway,
        one_hot_encode_route,
        cyclical_encode_temporal,
        create_feature_array
    )
    print("[OK] Preprocessing functions imported")
    
    # Create mock data
    mock_data = pd.DataFrame({
        'arrival_time': pd.date_range('2025-01-01', periods=100, freq='5min'),
        'headway': np.random.uniform(2, 15, 100),
        'route_id': np.random.choice(['A', 'C', 'E'], 100),
        'track': ['A1'] * 100,
        'time_of_day': np.random.randint(0, 24, 100),
        'day_of_week': np.random.randint(0, 7, 100)
    })
    print("[OK] Created mock data: {} samples".format(len(mock_data)))
    
    # Test log transform
    mock_data, stats = log_transform_headway(mock_data)
    print("[OK] Log transformation works")
    
    # Test one-hot encoding
    mock_data = one_hot_encode_route(mock_data)
    print("[OK] One-hot encoding works")
    
    # Test cyclical encoding
    mock_data = cyclical_encode_temporal(mock_data)
    print("[OK] Cyclical encoding works")
    
    # Test feature array creation
    X = create_feature_array(mock_data)
    print("[OK] Feature array created: {}".format(X.shape))
    print("  Expected shape: (100, 8)")
    assert X.shape == (100, 8), "Shape mismatch: {} vs (100, 8)".format(X.shape)
    print("[OK] Preprocessing test passed\n")
except Exception as e:
    print("[FAIL] Preprocessing test failed: {}\n".format(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Extract data (mock test - don't actually query BigQuery)
print("Test 4: Extract Data Module")
print("-" * 40)
try:
    import extract_data
    print("[OK] Extract data module imported")
    print("  Note: Skipping BigQuery connection test (requires credentials)")
    print("[OK] Extract data test passed\n")
except Exception as e:
    print("[FAIL] Extract data test failed: {}\n".format(e))
    sys.exit(1)

# Test 5: Training module imports
print("Test 5: Training Module")
print("-" * 40)
try:
    from train import (
        load_preprocessed_data,
        calculate_split_indices,
        create_callbacks
    )
    print("[OK] Training functions imported")
    
    # Test split calculation
    train_end, val_end, test_end = calculate_split_indices(1000)
    print("[OK] Split indices calculated: train={}, val={}, test={}".format(train_end, val_end, test_end))
    assert train_end == 600, "Train split incorrect"
    assert val_end == 800, "Val split incorrect"
    assert test_end == 1000, "Test split incorrect"
    print("[OK] Training test passed\n")
except Exception as e:
    print("[FAIL] Training test failed: {}\n".format(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Evaluation module imports
print("Test 6: Evaluation Module")
print("-" * 40)
try:
    from evaluate import (
        inverse_transform_headway,
        evaluate_classification,
        evaluate_regression
    )
    print("[OK] Evaluation functions imported")
    
    # Test inverse transform
    log_headways = np.log(np.array([5.0, 10.0, 15.0]) + config.LOG_OFFSET)
    headways = inverse_transform_headway(log_headways)
    print("[OK] Inverse transform works")
    print("[OK] Evaluation test passed\n")
except Exception as e:
    print("[FAIL] Evaluation test failed: {}\n".format(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("="*80)
print("All Module Tests Passed!")
print("="*80)
print()
print("Next steps:")
print("  1. Ensure BigQuery credentials are configured")
print("  2. Run: python extract_data.py")
print("  3. Run: python preprocess.py")
print("  4. Run: python train.py --run_name test_run")
print("  5. Run: python evaluate.py --run_name test_run")
print()
