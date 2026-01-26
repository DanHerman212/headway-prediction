import os
import pandas as pd
import numpy as np
import subprocess
import pytest
import shutil

def run_command(cmd, env_vars=None):
    """Run a shell command with specific environment variables."""
    # Copy current environment
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    # Add current directory to PYTHONPATH so 'src' module is found
    env['PYTHONPATH'] = os.getcwd()
    
    print(f"Running: {cmd}")
    process = subprocess.run(
        cmd, 
        shell=True, 
        env=env, 
        check=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    print(process.stdout)
    return process

def create_raw_data(path, n_rows=500):
    """Creates synthetic raw data matching BigQuery schema."""
    print(f"Generating {n_rows} rows of raw data at {path}...")
    
    # 2025-01-01 08:00:00
    base_time = pd.Timestamp("2025-01-01 08:00:00")
    
    times = [base_time + pd.Timedelta(seconds=i*300) for i in range(n_rows)]
    
    df = pd.DataFrame({
        'arrival_time': times,
        'route_id': np.random.choice(['A', 'C', 'E'], size=n_rows),
        'headway': np.random.lognormal(mean=1.5, sigma=0.5, size=n_rows),
        'time_of_day_seconds': [(t.hour * 3600 + t.minute * 60 + t.second) for t in times]
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print("Raw data generated.")

def test_pipeline_smoke():
    """Final Smoke Test for Production Workflow."""
    
    # Paths
    base_dir = "tests/artifacts/smoke"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    raw_path = f"{base_dir}/raw_data.csv"
    processed_path = f"{base_dir}/processed.csv"
    model_path = f"{base_dir}/model.keras"
    test_data_path = f"{base_dir}/test_set.csv"
    metrics_path = f"{base_dir}/metrics.json"
    html_path = f"{base_dir}/report.html"
    
    # Env for speed
    smoke_env = {
        "EPOCHS": "1",
        "BATCH_SIZE": "8",
        "LOOKBACK_STEPS": "10",
        "GCP_PROJECT_ID": "smoke-test-project"
    }
    
    try:
        # Step 1: Mock Ingestion
        create_raw_data(raw_path)
        
        # Step 2: Preprocess
        # NOTE: This will fail to find GTFS in GCS/Cache and fall back to dummy baseline, which is expected behavior
        cmd_prep = f"python3 src/preprocess.py --input_path {raw_path} --output_path {processed_path}"
        run_command(cmd_prep, smoke_env)
        
        assert os.path.exists(processed_path), "Preprocessing failed to create output file."
        
        # Step 3: Train
        # This produces model.keras AND test_set.csv
        cmd_train = f"python3 src/train.py --input_path {processed_path} --model_output_path {model_path} --test_data_output_path {test_data_path}"
        run_command(cmd_train, smoke_env)
        
        assert os.path.exists(model_path), "Training failed to create model."
        assert os.path.exists(test_data_path), "Training failed to export test set."
        
        # Step 4: Evaluate
        cmd_eval = f"python3 src/eval.py --model_path {model_path} --test_data_path {test_data_path} --output_metrics_path {metrics_path} --output_html_path {html_path}"
        run_command(cmd_eval, smoke_env)
        
        assert os.path.exists(metrics_path), "Evaluation failed to create metrics."
        assert os.path.exists(html_path), "Evaluation failed to create HTML report."
        
        print("\n✅ SMOKE TEST PASSED: Full pipeline executed successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ SMOKE TEST FAILED in step: {e.cmd}")
        print("STDERR:")
        print(e.stderr)
        raise e
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        raise e

if __name__ == "__main__":
    test_pipeline_smoke()
