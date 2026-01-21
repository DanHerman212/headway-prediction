#!/usr/bin/env python3
"""
Test Vertex AI Experiments and TensorBoard integration.
Run this BEFORE deploying to validate the integration works.

Usage:
    python test_vertex_integration.py
"""

import os
import sys
from datetime import datetime

# Set environment variables
os.environ['GCP_PROJECT_ID'] = 'realtime-headway-prediction'
os.environ['GCP_REGION'] = 'us-east1'
os.environ['GCS_BUCKET'] = 'ml-pipelines-headway-prediction'
os.environ['EXPERIMENT_NAME'] = 'a1-headway-prediction'
os.environ['BQ_DATASET'] = 'headway_prediction'
os.environ['BQ_TABLE'] = 'ml'

from src.config import config

print("="*80)
print("VERTEX AI INTEGRATION TEST")
print("="*80)
print(f"\nProject: {config.BQ_PROJECT}")
print(f"Region: {config.BQ_LOCATION}")
print(f"Experiment: {config.EXPERIMENT_NAME}")
print(f"GCS Bucket: {config.GCS_BUCKET}")

# Test 1: Import check
print("\n" + "="*80)
print("TEST 1: Checking imports")
print("="*80)
try:
    from google.cloud import aiplatform
    print("✓ google.cloud.aiplatform imported successfully")
except ImportError as e:
    print(f"✗ FAILED: Cannot import aiplatform: {e}")
    sys.exit(1)

# Test 2: Initialize Vertex AI
print("\n" + "="*80)
print("TEST 2: Initializing Vertex AI")
print("="*80)
try:
    aiplatform.init(
        project=config.BQ_PROJECT,
        location=config.BQ_LOCATION
    )
    print("✓ Vertex AI initialized successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Get or create experiment
print("\n" + "="*80)
print("TEST 3: Getting/Creating Experiment")
print("="*80)
try:
    # Try to get existing experiment
    try:
        experiment = aiplatform.Experiment(config.EXPERIMENT_NAME)
        print(f"✓ Found existing experiment: {config.EXPERIMENT_NAME}")
    except:
        # Create new experiment
        experiment = aiplatform.Experiment.create(
            experiment_name=config.EXPERIMENT_NAME,
            description="A1 Track Headway Prediction Model Training"
        )
        print(f"✓ Created new experiment: {config.EXPERIMENT_NAME}")
    
    print(f"  Resource name: {experiment.resource_name}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Start a test run
print("\n" + "="*80)
print("TEST 4: Starting test run")
print("="*80)
test_run_name = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
try:
    aiplatform.start_run(run=test_run_name, resume=False)
    print(f"✓ Started run: {test_run_name}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Log test parameters
print("\n" + "="*80)
print("TEST 5: Logging parameters")
print("="*80)
try:
    test_params = {
        'test_param_1': 'value1',
        'test_param_2': 42,
        'test_param_3': 3.14
    }
    aiplatform.log_params(test_params)
    print(f"✓ Logged {len(test_params)} parameters")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Log test metrics
print("\n" + "="*80)
print("TEST 6: Logging metrics")
print("="*80)
try:
    test_metrics = {
        'test_loss': 0.5,
        'test_accuracy': 0.85,
        'test_mae': 12.3
    }
    aiplatform.log_metrics(test_metrics, step=0)
    print(f"✓ Logged {len(test_metrics)} metrics")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Check TensorBoard instances
print("\n" + "="*80)
print("TEST 7: Checking TensorBoard instances")
print("="*80)
try:
    tensorboards = aiplatform.Tensorboard.list(
        filter=f'display_name="A1 Headway Prediction"',
        order_by='create_time desc'
    )
    if tensorboards:
        tb = tensorboards[0]
        print(f"✓ Found TensorBoard instance: {tb.display_name}")
        print(f"  Resource name: {tb.resource_name}")
    else:
        print("⚠ No TensorBoard instance found with name 'A1 Headway Prediction'")
        print("  You may need to create one manually in the Vertex AI console")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 8: End the run
print("\n" + "="*80)
print("TEST 8: Ending test run")
print("="*80)
try:
    aiplatform.end_run()
    print(f"✓ Ended run: {test_run_name}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print(f"\nYour experiment is ready: {config.EXPERIMENT_NAME}")
print(f"Test run created: {test_run_name}")
print(f"\nView in console:")
print(f"  Experiments: https://console.cloud.google.com/vertex-ai/experiments/experiments?project={config.BQ_PROJECT}")
print(f"  Your test run: https://console.cloud.google.com/vertex-ai/experiments/experiments/{config.EXPERIMENT_NAME}?project={config.BQ_PROJECT}")
print("\nYou can now deploy with confidence: ./deploy.sh baseline2")
print("="*80)
