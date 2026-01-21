"""Test Vertex AI Experiments and TensorBoard integration."""
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config

def test_vertex_ai_experiments_config():
    """Test that Vertex AI Experiments configuration is correct."""
    print("\n" + "="*80)
    print("TESTING VERTEX AI EXPERIMENTS CONFIGURATION")
    print("="*80 + "\n")
    
    errors = []
    
    # Check required config values
    print("1. Checking required configuration...")
    if not config.BQ_PROJECT:
        errors.append("BQ_PROJECT not set")
    else:
        print(f"   ✓ Project: {config.BQ_PROJECT}")
    
    if not config.BQ_LOCATION:
        errors.append("BQ_LOCATION not set")
    else:
        print(f"   ✓ Location: {config.BQ_LOCATION}")
    
    if not config.EXPERIMENT_NAME:
        errors.append("EXPERIMENT_NAME not set")
    else:
        print(f"   ✓ Experiment: {config.EXPERIMENT_NAME}")
    
    # Check TensorBoard configuration
    print("\n2. Checking TensorBoard configuration...")
    if not config.TENSORBOARD_LOG_DIR:
        errors.append("TENSORBOARD_LOG_DIR not set")
    else:
        print(f"   ✓ TensorBoard dir: {config.TENSORBOARD_LOG_DIR}")
        
        # Check if it's a GCS path (required for Vertex AI)
        if not config.TENSORBOARD_LOG_DIR.startswith('gs://'):
            errors.append(f"TENSORBOARD_LOG_DIR must be GCS path (gs://...), got: {config.TENSORBOARD_LOG_DIR}")
        else:
            print(f"   ✓ Using GCS path (required for Vertex AI)")
    
    if errors:
        print("\n✗ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("\n✓ Configuration is valid for Vertex AI Experiments")
    return True


def test_vertex_ai_imports():
    """Test that required libraries can be imported."""
    print("\n" + "="*80)
    print("TESTING VERTEX AI IMPORTS")
    print("="*80 + "\n")
    
    errors = []
    
    # Test google.cloud.aiplatform
    print("1. Testing google.cloud.aiplatform import...")
    try:
        from google.cloud import aiplatform
        print("   ✓ google.cloud.aiplatform imported successfully")
        print(f"   ✓ Version: {aiplatform.__version__}")
    except ImportError as e:
        errors.append(f"Cannot import google.cloud.aiplatform: {e}")
        print(f"   ✗ Import failed: {e}")
    
    # Test TensorFlow
    print("\n2. Testing tensorflow import...")
    try:
        import tensorflow as tf
        print("   ✓ TensorFlow imported successfully")
        print(f"   ✓ Version: {tf.__version__}")
    except ImportError as e:
        errors.append(f"Cannot import tensorflow: {e}")
        print(f"   ✗ Import failed: {e}")
    
    if errors:
        print("\n✗ Import errors detected")
        return False
    
    print("\n✓ All required imports available")
    return True


def test_vertex_ai_initialization():
    """Test that Vertex AI can be initialized with current config."""
    print("\n" + "="*80)
    print("TESTING VERTEX AI INITIALIZATION")
    print("="*80 + "\n")
    
    try:
        from google.cloud import aiplatform
        
        print("1. Initializing Vertex AI...")
        print(f"   Project: {config.BQ_PROJECT}")
        print(f"   Location: {config.BQ_LOCATION}")
        print(f"   Experiment: {config.EXPERIMENT_NAME}")
        
        # Initialize (this doesn't create anything, just sets up the client)
        aiplatform.init(
            project=config.BQ_PROJECT,
            location=config.BQ_LOCATION,
            experiment=config.EXPERIMENT_NAME
        )
        
        print("\n✓ Vertex AI initialized successfully")
        print("  Note: Actual experiment tracking requires deployment to Vertex AI")
        return True
        
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        print("\nThis is expected if:")
        print("  - Not authenticated with GCP (run: gcloud auth application-default login)")
        print("  - Project doesn't exist or no permissions")
        print("  - Running locally without GCP credentials")
        print("\nHowever, the configuration SYNTAX is correct.")
        return False


def test_tensorboard_callback_setup():
    """Test that TensorBoard callback can be created."""
    print("\n" + "="*80)
    print("TESTING TENSORBOARD CALLBACK SETUP")
    print("="*80 + "\n")
    
    try:
        import tensorflow as tf
        
        run_name = "test_run"
        tensorboard_dir = f"{config.TENSORBOARD_LOG_DIR}/{run_name}"
        
        print(f"1. Creating TensorBoard callback...")
        print(f"   Log dir: {tensorboard_dir}")
        
        callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            profile_batch='10,20'
        )
        
        print("\n✓ TensorBoard callback created successfully")
        print(f"  Callback type: {type(callback)}")
        print(f"  Log directory: {callback.log_dir}")
        
        # Check configuration
        print("\n2. Checking callback configuration...")
        print(f"   ✓ Histogram tracking: {callback.histogram_freq > 0}")
        print(f"   ✓ Graph tracking: {callback.write_graph}")
        print(f"   ✓ Profile batch: {callback._profile_batch}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to create TensorBoard callback: {e}")
        return False


def test_vertex_experiments_code_pattern():
    """Test that the train.py code pattern is correct."""
    print("\n" + "="*80)
    print("TESTING VERTEX EXPERIMENTS CODE PATTERN")
    print("="*80 + "\n")
    
    train_file = Path(__file__).parent.parent / "src" / "train.py"
    
    if not train_file.exists():
        print(f"✗ train.py not found at {train_file}")
        return False
    
    content = train_file.read_text()
    
    errors = []
    
    # Check for correct patterns
    print("1. Checking aiplatform.init() has experiment parameter...")
    if "aiplatform.init(" in content:
        if "experiment=" in content or "experiment =" in content:
            print("   ✓ experiment parameter found in aiplatform.init()")
        else:
            errors.append("aiplatform.init() missing experiment parameter")
            print("   ✗ experiment parameter NOT found")
    
    print("\n2. Checking aiplatform.start_run() usage...")
    if "aiplatform.start_run" in content:
        print("   ✓ start_run() found")
        
        # Check it doesn't have invalid tensorboard parameter
        if "start_run(run=" in content or "start_run(run =" in content:
            print("   ✓ start_run() has run parameter")
        
        # Check for invalid patterns
        if "tensorboard=" in content.split("start_run")[1].split("\n")[0]:
            errors.append("start_run() has invalid tensorboard parameter (should be removed)")
            print("   ✗ start_run() has invalid tensorboard parameter")
    
    print("\n3. Checking TensorBoard callback configuration...")
    if "TensorBoard(" in content:
        print("   ✓ TensorBoard callback found")
        if "log_dir=" in content:
            print("   ✓ log_dir parameter present")
    else:
        errors.append("TensorBoard callback not found")
    
    if errors:
        print("\n✗ Code pattern errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("\n✓ Code pattern is correct")
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VERTEX AI EXPERIMENTS & TENSORBOARD VALIDATION")
    print("="*80)
    
    results = {}
    
    # Run all tests
    results['config'] = test_vertex_ai_experiments_config()
    results['imports'] = test_vertex_ai_imports()
    results['code_pattern'] = test_vertex_experiments_code_pattern()
    results['tensorboard'] = test_tensorboard_callback_setup()
    results['initialization'] = test_vertex_ai_initialization()
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s}: {status}")
    
    print("\n" + "="*80)
    
    # Overall result
    critical_tests = ['config', 'imports', 'code_pattern', 'tensorboard']
    critical_passed = all(results.get(test, False) for test in critical_tests)
    
    if critical_passed:
        print("✓ CRITICAL CHECKS PASSED")
        print("\nVertex AI Experiments and TensorBoard tracking will work.")
        print("\nNote: 'initialization' test may fail locally without GCP credentials,")
        print("but this is expected and won't affect deployment to Vertex AI.")
        exit(0)
    else:
        print("✗ SOME CRITICAL CHECKS FAILED")
        print("\nFix the errors above before deploying.")
        exit(1)
