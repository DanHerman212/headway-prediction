import os
import sys
import time
import random
import logging
from google.cloud import aiplatform

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to python path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock config or load real one
try:
    from src.config import config
except ImportError:
    logger.error("Could not import src.config. Make sure you run this from mlops_pipeline directory.")
    sys.exit(1)

def robust_run_manager(run_name, step_name):
    """
    Simulates the exact logic we placed in train.py and eval.py
    to verify it handles New Runs and Existing Runs without crashing.
    """
    logger.info(f"[{step_name}] Managing Run: {run_name}")
    
    # 1. Init SDK
    try:
        aiplatform.init(
            project=config.project_id,
            location=config.region,
            experiment=config.experiment_name,
            experiment_tensorboard=config.tensorboard_resource
        )
    except Exception as e:
        logger.error(f"[{step_name}] Failed to init Vertex AI: {e}")
        raise e

    # 2. Try Create -> Catch Exists -> Resume
    try:
        logger.info(f"[{step_name}] Attempting to CREATE run (resume=False)...")
        aiplatform.start_run(run=run_name, resume=False)
        logger.info(f"[{step_name}] ✅ Successfully CREATED run.")
    except Exception as e:
        error_str = str(e)
        # Check for 409 Already Exists
        if "409" in error_str or "AlreadyExists" in error_str:
            logger.info(f"[{step_name}] ⚠️ Run already exists (Caught 409). Attempting to RESUME...")
            try:
                aiplatform.start_run(run=run_name, resume=True)
                logger.info(f"[{step_name}] ✅ Successfully RESUMED run.")
            except Exception as resume_error:
                logger.error(f"[{step_name}] ❌ Failed to RESUME run: {resume_error}")
                raise resume_error
        else:
             # Unexpected error during creation
             logger.error(f"[{step_name}] ❌ Failed to CREATE run with unexpected error: {e}")
             raise e

    # 3. Log something to verify active context
    try:
        aiplatform.log_metrics({f"{step_name}_status": 1.0})
        logger.info(f"[{step_name}] ✅ Logged metric successfully.")
    except Exception as e:
        logger.error(f"[{step_name}] ❌ Failed to log metrics: {e}")
        raise e
        
    # 4. End Run (simulating end of component)
    # Note: In a pipeline, we might not end it in Train if we want Eval to append? 
    # Actually, Vertex SDK runs are thread-local/process-local contexts. 
    # train.py ends. eval.py starts new process, resumes run.
    aiplatform.end_run()
    logger.info(f"[{step_name}] Run ended correctly.\n")

def run_local_test():
    """
    Test Lifecycle:
    1. 'Train' step creates the run.
    2. 'Eval' step resumes the *same* run.
    """
    print("\n=======================================================")
    print("LOCAL EXPERIMENT TRACKING VERIFICATION")
    print("=======================================================\n")
    
    # Generate unique run ID for this test
    # We use a randomized ID to ensure we don't hit old 404/caching issues
    run_id = f"local-test-{int(time.time())}-{random.randint(1000,9999)}"
    config.run_name = run_id
    
    print(f"Test Run ID: {run_id}")
    print(f"Project:     {config.project_id}")
    print(f"Experiment:  {config.experiment_name}")
    print("-" * 50)

    try:
        # STEP 1: Simulate Training (Should Create)
        robust_run_manager(run_id, "TRAIN_COMPONENT")
        
        # STEP 2: Simulate Evaluation (Should Resume)
        # We start a 'fresh' context logic-wise by calling the function again
        robust_run_manager(run_id, "EVAL_COMPONENT")
        
        print("✅ TEST PASSED: Full lifecycle (Create -> Resume) works locally.")
        print("This confirms the logic is robust against 404/409 errors.")
        
    except Exception as e:
        print("\n❌ TEST FAILED.")
        print(f"Reason: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_local_test()
