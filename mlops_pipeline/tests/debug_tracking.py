import os
import sys
import time
from google.cloud import aiplatform

# Add project root to python path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config

def debug_experiment_tracking():
    print("--- Debugging Experiment Tracking ---")
    
    # 1. Check Config
    print(f"Project ID: {config.project_id}")
    print(f"Region: {config.region}")
    print(f"Experiment Name: {config.experiment_name}")
    print(f"TensorBoard: {config.tensorboard_resource}")
    
    # Generate a unique run name for this test
    test_run_name = f"debug-run-{int(time.time())}"
    config.run_name = test_run_name # Override config
    print(f"Test Run Name: {test_run_name}")

    # 2. Initialize Vertex AI
    print("\n[Action] Initializing Vertex AI SDK...")
    try:
        aiplatform.init(
            project=config.project_id,
            location=config.region,
            experiment=config.experiment_name,
            experiment_tensorboard=config.tensorboard_resource
        )
        print("✅ aiplatform.init() successful")
    except Exception as e:
        print(f"❌ aiplatform.init() failed: {e}")
        return

    # 3. Start Run (Resume=True) - Verification of "Resuming nonexistent run" behavior
    print(f"\n[Action] Attempting start_run(run='{test_run_name}', resume=True)...")
    try:
        aiplatform.start_run(run=test_run_name, resume=True)
        print("✅ start_run(resume=True) succeeded (unexpected for new run, but maybe it auto-creates?)")
    except Exception as e:
        print(f"❌ start_run(resume=True) failed as expected for new run: {e}")
        
        # 4. Start Run (Resume=False) - Creation
        print(f"\n[Action] Attempting start_run(run='{test_run_name}', resume=False)...")
        try:
            aiplatform.start_run(run=test_run_name, resume=False)
            print("✅ start_run(resume=False) succeeded. Run created.")
        except Exception as e2:
            print(f"❌ start_run(resume=False) failed: {e2}")
            return

    # 5. Log Metrics
    print("\n[Action] Logging dummy metrics...")
    try:
        aiplatform.log_metrics({"test_accuracy": 0.99, "test_loss": 0.01})
        print("✅ log_metrics() succeeded")
    except Exception as e:
        print(f"❌ log_metrics() failed: {e}")

    # 6. Log Params
    print("\n[Action] Logging dummy params...")
    try:
        aiplatform.log_params({"learning_rate": 0.001, "batch_size": 32})
        print("✅ log_params() succeeded")
    except Exception as e:
        print(f"❌ log_params() failed: {e}")

    # 7. End Run
    print("\n[Action] Ending run...")
    try:
        aiplatform.end_run()
        print("✅ end_run() succeeded")
    except Exception as e:
        print(f"❌ end_run() failed: {e}")

if __name__ == "__main__":
    debug_experiment_tracking()
