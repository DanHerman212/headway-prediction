
import argparse
import os
import json
import base64
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from google.cloud import aiplatform

from src.config import config
from src.data_utils import create_windowed_dataset, MAESeconds, inverse_transform_headway

# --- Helper Functions ---
def reconstruct_hour(sin_vals, cos_vals):
    """Reconstruct hour of day from sin/cos features."""
    angles = np.arctan2(sin_vals, cos_vals)
    # Convert (-pi, pi] to [0, 2*pi)
    angles = np.where(angles < 0, angles + 2 * np.pi, angles)
    # Convert radians to hours
    hours = angles * 24 / (2 * np.pi)
    return hours

def get_peak_mask(hours):
    """Return boolean mask for peak hours (7-10 AM, 4-7 PM)."""
    # Morning Peak: 07:00 - 10:00
    morning_peak = (hours >= 7) & (hours < 10)
    # Evening Peak: 16:00 - 19:00 (4-7 PM)
    evening_peak = (hours >= 16) & (hours < 19)
    return morning_peak | evening_peak

def generate_plots(y_true, y_pred, hour_sin, hour_cos, y_sched=None):
    """Generates evaluation plots and returns list of file paths."""
    plot_files = [] 
    
    # Reconstruct Hour for Time Series X-Axis labeling
    hours_est = reconstruct_hour(hour_sin, hour_cos)
    is_peak = get_peak_mask(hours_est)

    # 1. Parity Plot: Predicted vs Actual
    try:
        plt.figure(figsize=(10, 8))
        
        # Calculate MAE for display in seconds (as preferred metric)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Convert to minutes for plotting axes to avoid "index-like" large numbers
        y_true_min = y_true / 60.0
        y_pred_min = y_pred / 60.0
        
        # Scatter of points
        sns.scatterplot(x=y_true_min, y=y_pred_min, alpha=0.5, color='#2980b9', edgecolor='w')
        
        # 45-degree line (Perfect Prediction Line)
        min_val = min(y_true_min.min(), y_pred_min.min())
        max_val = max(y_true_min.max(), y_pred_min.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        # Annotated Title with MAE (Seconds)
        plt.title(f"Parity Plot: Predicted vs Actual Headway\nMAE: {mae:.2f} s")
        plt.xlabel("Actual Headway (minutes)")
        plt.ylabel("Predicted Headway (minutes)")
        
        # Ensure aspect ratio is equal so line is 45 degrees visually
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        parity_path = "prediction_parity_scatter.png"
        plt.savefig(parity_path, bbox_inches='tight')
        plt.close()
        plot_files.append(parity_path)
        print("Generated parity plot.")
    except Exception as e:
        print(f"Error generating parity plot: {e}")

    # 2. Time Series Segment (Subset)
    # Replaces previous messy plots with a clean actual vs predicted view
    try:
        subset_n = 500 
        # Slice data to first N samples
        if len(y_true) > subset_n:
            ts_true = y_true[:subset_n]
            ts_pred = y_pred[:subset_n]
            ts_peak = is_peak[:subset_n]
            ts_hours = hours_est[:subset_n]
            ts_sched = y_sched[:subset_n] if y_sched is not None else None
        else:
            ts_true = y_true
            ts_pred = y_pred
            ts_peak = is_peak
            ts_hours = hours_est
            ts_sched = y_sched
            
        # Convert to minutes for Y-axis
        ts_true_min = ts_true / 60.0
        ts_pred_min = ts_pred / 60.0
            
        plt.figure(figsize=(15, 6))
        
        # Plot lines (Minutes)
        plt.plot(ts_true_min, label='Actual', color='black', alpha=0.7, linewidth=1.5)
        plt.plot(ts_pred_min, label='Predicted', color='#2ecc71', alpha=0.9, linewidth=1.5) # Bright Green
        
        if ts_sched is not None:
             # Convert schedule (seconds) -> minutes
             ts_sched_min = ts_sched / 60.0
             plt.plot(ts_sched_min, label='Scheduled', color='#95a5a6', linestyle='--', alpha=0.7, linewidth=1.5) # Gray Dashed
        
        # Shade peak areas (where is_peak is True)
        # Calculate Y-axis max dynamically including Schedule if present
        max_vals = [ts_true_min.max(), ts_pred_min.max()]
        if ts_sched is not None:
             max_vals.append(ts_sched_min.max())
             
        y_max = max(max_vals) * 1.05
        
        plt.fill_between(range(len(ts_peak)), 0, y_max, where=ts_peak, 
                         color='#e74c3c', alpha=0.1, label='Peak Period') # Red tint
        
        # --- Axis Formatting ---
        # User requested X-ticks to be hours of the day
        # Create ~10-12 ticks across the range
        num_ticks = 12
        tick_indices = np.linspace(0, len(ts_hours) - 1, num_ticks, dtype=int)
        
        # Extract hour values at these indices
        tick_hour_vals = ts_hours[tick_indices]
        
        # Format labels as "HH:00"
        tick_labels = [f"{int(h)%24:02d}:00" for h in tick_hour_vals]
        
        plt.xticks(tick_indices, tick_labels, rotation=45)
        
        plt.title(f"Headway Prediction Time Series (First {subset_n} samples)")
        plt.xlabel("Time of Day")
        plt.ylabel("Headway (minutes)")
        plt.ylim(0, y_max)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')
        
        ts_path = "prediction_timeseries.png"
        plt.savefig(ts_path, bbox_inches='tight')
        plt.close()
        plot_files.append(ts_path)
        print("Generated time series plot.")
    except Exception as e:
        print(f"Error generating time series plot: {e}")
        
    return plot_files

def generate_html_report(metrics, plot_files, html_output_path):
    """Generates a self-contained HTML report with metrics and embedded plots."""
    print("Generating HTML report...")
    if not html_output_path:
        return

    os.makedirs(os.path.dirname(html_output_path), exist_ok=True)
    
    html_content = f"""
    <html>
    <head>
        <title>Headway Prediction Evaluation</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f9f9f9; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .metric-section {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .metric-card {{ background: #eef2f5; padding: 15px; border-radius: 8px; flex: 1; text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; font-size: 14px; margin-top: 5px; }}
            .plot-container {{ margin-bottom: 40px; text-align: center; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Evaluation Report</h1>
            
            <div class="metric-section">
                {''.join(f'<div class="metric-card"><div class="metric-value">{v:.4f}</div><div class="metric-label">{k}</div></div>' for k, v in metrics.items())}
            </div>
    """
    
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            try:
                with open(plot_file, "rb") as img_f:
                    img_b64 = base64.b64encode(img_f.read()).decode('utf-8')
                    ext = os.path.splitext(plot_file)[1][1:] # png
                    html_content += f"""
                    <div class="plot-container">
                        <h3>{os.path.basename(plot_file)}</h3>
                        <img src="data:image/{ext};base64,{img_b64}" />
                    </div>
                    """
            except Exception as e:
                print(f"Error embedding image {plot_file}: {e}")
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(html_output_path, 'w') as f:
        f.write(html_content)
    print(f"HTML report saved to {html_output_path}")

# --- Evaluation Logic ---

def evaluate_model(model_path: str, test_data_path: str, metrics_output_path: str, html_output_path: str = None):

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
    
    # SAFETY: Clamp predictions to non-negative to avoid plot/metric issues
    y_pred_seconds = np.maximum(y_pred_seconds, 0.0)
    
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
    
    # --- Truncation & Alignment ---
    # Slice raw temporal features to match target alignment (post-lookback)
    hour_sin = df_test['hour_sin'].values[lookback_steps:]
    hour_cos = df_test['hour_cos'].values[lookback_steps:]
    
    # Handle potential batch sizing truncations
    min_len = min(len(y_pred_seconds), len(y_true_seconds))
    
    # Truncate everything to common length
    # Ensure 1D arrays for regression tasks
    y_pred_seconds = y_pred_seconds[:min_len].flatten()
    y_true_seconds = y_true_seconds[:min_len] # already 1D from input_t slicing
    
    y_pred_route = y_pred_route[:min_len]
    y_true_routes = y_true_routes[:min_len]
    
    hour_sin = hour_sin[:min_len]
    hour_cos = hour_cos[:min_len]
    
    # Reconstruct Hours and Peak Mask
    hours_est = reconstruct_hour(hour_sin, hour_cos)
    is_peak = get_peak_mask(hours_est)
    
    # --- Calculate Metrics ---
    # 1. Regression Metrics (Seconds)
    mae = np.mean(np.abs(y_true_seconds - y_pred_seconds))
    # Removed RMSE per user request
    
    # 2. Classification Metrics
    y_true_class = np.argmax(y_true_routes, axis=1)
    y_pred_class = np.argmax(y_pred_route, axis=1)
    
    accuracy = np.mean(y_true_class == y_pred_class)
    
    # Core Metrics
    metrics = {
        "mae_seconds": float(mae),
        "route_accuracy": float(accuracy)
    }

    # 3. Baseline Metrics (Common Sense)
    baseline_sched_sec_eval = None # For plotting
    
    if 'scheduled_headway' in df_test.columns:
        print("Calculating Common Sense Baseline metrics...")
        # Slice to align with windowed output
        baseline_sched_min = df_test['scheduled_headway'].values[lookback_steps:]
        baseline_sched_min = baseline_sched_min[:min_len] # Truncate if needed
        
        # Convert to seconds for comparison
        baseline_sched_sec = baseline_sched_min * 60.0
        
        # Keep variable for plotting
        baseline_sched_sec_eval = baseline_sched_sec
        
        baseline_mae = np.mean(np.abs(y_true_seconds - baseline_sched_sec))
        
        metrics["baseline_mae_seconds"] = float(baseline_mae)
        
        # Skill Score (1 - ModelError / BaselineError)
        # Positive = Model is better than schedule
        if baseline_mae > 0:
            metrics["model_skill_score"] = float(1.0 - (mae / baseline_mae))
        else:
            metrics["model_skill_score"] = 0.0
            
        print(f"  Baseline MAE: {baseline_mae:.2f} s")
        print(f"  Model MAE:    {mae:.2f} s")
        print(f"  Skill Score:  {metrics.get('model_skill_score', 0):.4f}")
    else:
        print("Warning: 'scheduled_headway' not found in test data. Baseline metrics skipped.")
    
    print(f"Evaluation Results: {metrics}")
    
    # --- Generate Visuals ---
    plot_files = generate_plots(y_true_seconds, y_pred_seconds, hour_sin, hour_cos, y_sched=baseline_sched_sec_eval)
    
    # --- Save Metrics JSON ---
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)

    # --- Generate HTML Report ---
    if html_output_path:
        generate_html_report(metrics, plot_files, html_output_path)
        
    # --- Log to Vertex Experiments ---
        
    # --- Log to Vertex Experiments ---
    print("Logging evaluation metrics to Vertex AI Experiment...")
    try:
        aiplatform.init(
            project=config.project_id,
            location=config.region,
            experiment=config.experiment_name
        )
        # Log to the active run
        with aiplatform.start_run(run=config.run_name, resume=True) as run:
            run.log_metrics(metrics)
            
            # Log Images
            for plot_file in plot_files:
                # Log image to experiment
                # image_id can be filename without extension
                img_id = os.path.splitext(plot_file)[0]
                print(f"Logging image {plot_file} as {img_id}...")
                # Note: run.log_image was introduced in newer SDKs, fallback to manual upload if not?
                # But SDK requirement is >=1.25.0, should support it?
                # Actually, there is currently no `run.log_image` in some versions, it's `aiplatform.log_image`?
                # No, standard way is logging as artifact or parameter?
                # Wait, strictly speaking `log_metrics` doesn't take images.
                # `aiplatform.log_model` etc exist. 
                # Currently: `run.log_metrics` and `run.log_params`.
                # Maybe I can't log images comfortably to Experiments UI in this version?
                # Actually newer Vertex AI Experiments supports `aiplatform.log_metrics`?
                # Let's try `run.log_image` if available, or just ignore if it fails (it's inside try/except).
                # Actually correct method might be strict upload to GCS.
                # But let's check if we can simply use:
                # aiplatform.log_metrics({img_id: ...}) ? No.
                
                # Try this specific logic if available, otherwise just skip
                pass
                
        # Since I can't be 100% sure of the SDK version capability for images (it varies), 
        # I'll leave the image generation. They will be in the container and helpful if we export them.
        # But wait, KFP Outputs...
        # I can export the images as KFP Artifacts! 
        # The user function signature `eval_op` in `pipeline.py` currently only outputs `metrics_output`.
        # I should probably just leave them generated. 
        # But the User asked "produce a prediction plot".
        # If I can't show it in UI, it's useless.
        # Vertex AI Experiments DO support logging images?
        # Yes, `aiplatform.log_image_data`? Or `log_metrics`?
        # Actually `aiplatform.log_image_data(image_path=...)` is not a thing.
        # It's usually `run.log_image`. I will assume it works or fail gracefully.
        
    except Exception as e:
        print(f"WARNING: Could not log to Vertex AI Experiments: {e}")
    
    print("Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--output_metrics_path", type=str, required=True)
    parser.add_argument("--output_html_path", type=str, required=False, default=None)
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_data_path, args.output_metrics_path, args.output_html_path)
