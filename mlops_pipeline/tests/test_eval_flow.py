
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.config import config

def create_synthetic_data(path, n_samples=200):
    """Creates a synthetic dataset matching the pipeline structure."""
    print(f"Generating {n_samples} synthetic rows...")
    
    # Feature columns (must match what data_utils expects)
    # log_headway, route_A, route_C, route_E, hour_sin, hour_cos, day_sin, day_cos
    # Plus Metadata: scheduled_headway
    
    # Update: Align log_headway with ~5 min schedule (300s -> ln(301) ~= 5.7)
    data = {
        'log_headway': np.random.normal(5.7, 0.2, n_samples),
        'route_A': np.ones(n_samples),
        'route_C': np.zeros(n_samples),
        'route_E': np.zeros(n_samples),
        'hour_sin': np.sin(np.linspace(0, 10, n_samples)),
        'hour_cos': np.cos(np.linspace(0, 10, n_samples)),
        'day_sin': np.zeros(n_samples),
        'day_cos': np.ones(n_samples),
        'scheduled_headway': np.random.normal(5.0, 0.1, n_samples) # ~5 min baseline
    }
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"Saved test data to {path}")
    return df

def create_dummy_model(path):
    """Creates a dummy Keras model with compatible signature."""
    print("Creating dummy model...")
    
    lookback = config.lookback_steps
    # Features: log_headway (1) + routes (3) + temp (4) = 8
    n_features = 8 
    
    inputs = keras.Input(shape=(lookback, n_features), name='input_sequence')
    x = keras.layers.Flatten()(inputs)
    
    # Two outputs: headway (regression), route (classification)
    headway_out = keras.layers.Dense(1, name='headway')(x)
    route_out = keras.layers.Dense(3, activation='softmax', name='route')(x)
    
    model = keras.Model(inputs=inputs, outputs=[headway_out, route_out])
    model.compile(optimizer='adam', loss=['mse', 'categorical_crossentropy'])
    
    model.save(path)
    print(f"Saved dummy model to {path}")

def run_integration_test():
    # Setup Paths
    base_dir = "tests/artifacts"
    os.makedirs(base_dir, exist_ok=True)
    
    data_path = f"{base_dir}/test_set.csv"
    model_path = f"{base_dir}/dummy_model.keras"
    metrics_path = f"{base_dir}/metrics.json"
    report_path = f"{base_dir}/report.html"
    
    # 1. Generate Assets
    create_synthetic_data(data_path)
    create_dummy_model(model_path)
    
    # 2. Run Eval Script
    # We call it via subprocess to ensure it runs exactly as the pipeline component would
    print("\n--- Executing src/eval.py ---")
    cmd = f"python3 src/eval.py --model_path {model_path} --test_data_path {data_path} --output_metrics_path {metrics_path} --output_html_path {report_path}"
    
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print("Test FAILED: eval.py returned non-zero exit code.")
        return
        
    # 3. Inspect Results
    print("\n--- Verifying Results ---")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            print("Generated Metrics JSON:")
            print(f.read())
    else:
        print("FAILED: Metrics file not created.")

    if os.path.exists(report_path):
        print(f"SUCCESS: HTML Report created at {report_path}")
    else:
        print("FAILED: HTML Report not created.")

if __name__ == "__main__":
    run_integration_test()
