import os
import sys
import shutil
import pandas as pd
import numpy as np
import subprocess

def create_dummy_data(path):
    print(f"Creating dummy data at {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    N = 1000
    df = pd.DataFrame({
        'log_headway': np.random.randn(N),
        'route_A': np.random.randint(0, 2, N),
        'route_C': np.random.randint(0, 2, N),
        'route_E': np.random.randint(0, 2, N),
        # Add simpler features to match model expectation if needed, or rely on Trainer to handle it
        # Trainer.load_data: data.values, log_headway, route_*
        # Usually X is all columns.
    })
    # Trainer expects specific columns? 
    # trainer.load_data says:
    # self.input_x = self.data.values
    # self.input_t = self.data['log_headway'].values
    # so as long as 'log_headway' exists, it should be fine.
    
    # Fill unrelated columns to simulate features
    for i in range(5):
        df[f'feature_{i}'] = np.random.randn(N)
        
    df.to_csv(path, index=False)

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("COMMAND FAILED:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    else:
        print("COMMAND SUCCESS")
        print(result.stdout)

def main():
    # __file__ = .../ml_pipelines/scripts/verify_model_saving.py
    # scripts_dir = .../ml_pipelines/scripts
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    # package_root = .../ml_pipelines
    package_root = os.path.dirname(scripts_dir)
    # project_root = .../headway-prediction
    project_root = os.path.dirname(package_root)
    
    test_dir = os.path.join(package_root, "test_verify_io")
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    input_csv = os.path.join(test_dir, "X.csv")
    model_dir = os.path.join(test_dir, "model_output")
    test_data_path = os.path.join(test_dir, "test_dataset") # Directory for artifact
    eval_output = os.path.join(test_dir, "eval_output")
    metrics_output = os.path.join(test_dir, "metrics.json")
    tb_dir = os.path.join(test_dir, "logs")
    
    create_dummy_data(input_csv)
    
    # 1. Run Train
    print("\n--- TEST: TRAINING ---")
    train_cmd = [
        sys.executable, "-m", "ml_pipelines.training.train",
        "--input_csv", input_csv,
        "--model_dir", model_dir,
        "--test_dataset_path", test_data_path,
        "--epochs", "1",
        "--tensorboard_dir", tb_dir,
        "--tensorboard_resource_name", "projects/000/locations/us-central1/tensorboards/000"
    ]
    # Set env vars to avoid actual Vertex calls if possible, or expect errors in tracking but success in saving
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    env["USE_VERTEX_EXPERIMENTS"] = "false"
    env["GCP_PROJECT_ID"] = "dummy"
    env["VERTEX_LOCATION"] = "us-east1"


    # Mocking or trusting code handles errors gracefully
    
    # Running subprocess directly
    subprocess.run(train_cmd, env=env, check=True)
    
    # CHECK: Did it create model_output/saved_model.pb?
    expected_pb = os.path.join(model_dir, "saved_model.pb")
    if os.path.exists(expected_pb):
        print(f"SUCCESS: Found {expected_pb}")
    else:
        print(f"FAILURE: {expected_pb} NOT FOUND. Listing contents of {model_dir}:")
        for root, dirs, files in os.walk(model_dir):
            for name in files:
                print(os.path.join(root, name))
        sys.exit(1)

    # 2. Run Evaluate
    print("\n--- TEST: EVALUATION ---")
    eval_cmd = [
        sys.executable, "-m", "ml_pipelines.evaluation.evaluate_model",
        "--model", model_dir,
        "--data", test_data_path, # KFP often passes dir, code handles dir or file
        "--pre_split",
        "--output", eval_output,
        "--metrics_output", metrics_output
    ]
    
    subprocess.run(eval_cmd, env=env, check=True)
    
    print("\n--- VERIFICATION COMPLETE: ALL SYSTEMS GO ---")

if __name__ == "__main__":
    main()
