
import argparse
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from google.cloud import aiplatform

from src.config import config
from src.data_utils import create_windowed_dataset, MAESeconds, inverse_transform_headway

# --- Evaluation Logic ---

def evaluate_model(model_path: str, test_data_path: str, metrics_output_path: str):
    print(f"Loading test data from {test_data_path}...")
    df_test = pd.read_csv(test_data_path)
    print(f"Test Data Shape: {df_test.shape}")
    
    # --- Prepare X/y ---
    print("Preparing test features and targets...")
    
    # Use shared windowing logic
    # Note: create_windowed_dataset yields (x, y)
    # create_windowed_dataset shuffle defaults to True, we MUST set to False for evaluation alignment
    
    batch_size = config.batch_size
    lookback_steps = config.lookback_steps
    
    ds = create_windowed_dataset(
        df_test,
        batch_size=batch_size,
        lookback_steps=lookback_steps,
        start_index=0,
        end_index=None,
        shuffle=False  # CRITICAL: Do not shuffle test data so we can align it with ground truth
    )
    
    # --- Load Model ---
    print(f"Loading model from {model_path}...")
    # Load with custom object scope used in training
    with keras.utils.custom_object_scope({'MAESeconds': MAESeconds}):
        model = keras.models.load_model(model_path)
    
    print("Generating predictions...")
    results = model.predict(ds)
    
    # Results is a list: [headway_pred, route_pred] because it is a multi-output model
    y_pred_headway_log = results[0]
    y_pred_route = results[1]
    
    # --- Inverse Transform ---
    print("Inverse converting predictions (Log -> Seconds)...")
    y_pred_seconds = inverse_transform_headway(y_pred_headway_log)
    
    # --- Extract True Values ---
    # We need the true targets corresponding to the windows in `ds`.
    # `create_windowed_dataset` generates targets at index `i + lookback_steps`.
    # Range of i is 0 to (N - lookback_steps).
    # So targets are [lookback_steps, ..., N-1].
    
    # Targets (Log space) for full DF
    input_t = df_test['log_headway'].values
    route_cols = [c for c in df_test.columns if c.startswith('route_')]
    input_r = df_test[route_cols].values
    
    # Slicing to match windowed dataset output
    y_true_log = input_t[lookback_steps:]
    y_true_seconds = inverse_transform_headway(y_true_log)
    
    # Classification targets
    y_true_routes = input_r[lookback_steps:]
    
    # Handle potential batch sizing truncations (though model.predict usually handles last batch)
    # To be safe, truncate to the smaller length if they differ slightly
    min_len = min(len(y_pred_seconds), len(y_true_seconds))
    if len(y_pred_seconds) != len(y_true_seconds):
        print(f"INFO: Truncating arrays to length {min_len} due to mismatch.")
        y_pred_seconds = y_pred_seconds[:min_len]
        y_true_seconds = y_true_seconds[:min_len]
        y_pred_route = y_pred_route[:min_len]
        y_true_routes = y_true_routes[:min_len]
    
    # --- Calculate Metrics ---
    # 1. Regression Metrics (Seconds)
    mae = np.mean(np.abs(y_true_seconds - y_pred_seconds.flatten()))
    
    # RMSE calculation
    rmse = np.sqrt(np.mean((y_true_seconds - y_pred_seconds.flatten())**2))
    
    # 2. Classification Metrics
    y_true_class = np.argmax(y_true_routes, axis=1)
    y_pred_class = np.argmax(y_pred_route, axis=1)
    
    accuracy = np.mean(y_true_class == y_pred_class)
    
    metrics = {
        "mae_seconds": float(mae),
        "rmse_seconds": float(rmse),
        "route_accuracy": float(accuracy)
    }
    
    print(f"Evaluation Results: {metrics}")
    
    # --- Save Metrics ---
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)
        
    # --- Log to Vertex Experiments ---
    print("Logging evaluation metrics to Vertex AI Experiment...")
    try:
        aiplatform.init(
            project=config.project_id,
            location=config.region,
            experiment=config.experiment_name
        )
        # Log to the same run
        aiplatform.start_run(run=config.run_name, resume=True)
        aiplatform.log_metrics(metrics)
        aiplatform.end_run()
        print("Logged metrics successfully.")
    except Exception as e:
        print(f"WARNING: Could not log to Vertex AI Experiments: {e}")
    
    print("Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--output_metrics_path", type=str, required=True)
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_data_path, args.output_metrics_path)
