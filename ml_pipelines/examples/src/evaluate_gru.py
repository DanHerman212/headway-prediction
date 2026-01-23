import argparse
import pandas as pd
import numpy as np
import os
import json
import joblib
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # Set backend to Agg for headless environments
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error
from google.cloud import storage
import base64
from io import BytesIO

def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return tf.maximum(q * e, (q - 1) * e)

def combined_quantile_loss(y_true, y_pred):
    q1_pred = y_pred[:, 0:1]
    q5_pred = y_pred[:, 1:2]
    q9_pred = y_pred[:, 2:3]
    loss_q1 = tf.reduce_mean(quantile_loss(0.1, y_true, q1_pred))
    loss_q5 = tf.reduce_mean(quantile_loss(0.5, y_true, q5_pred))
    loss_q9 = tf.reduce_mean(quantile_loss(0.9, y_true, q9_pred))
    return loss_q1 + loss_q5 + loss_q9

def evaluate_gru(model_dir, test_ds):
    print("Evaluating GRU Model...")
    
    model_path = os.path.join(model_dir, "gru_model.keras")
    if os.path.exists(model_path):
            print(f"Loading Keras model from {model_path}...")
            model = tf.keras.models.load_model(model_path, custom_objects={'combined_quantile_loss': combined_quantile_loss})
    else:
            raise FileNotFoundError(f"Could not find valid model at {model_path}")

    # Predict
    print("Generating predictions...")
    predictions = model.predict(test_ds, verbose=1)
    
    # Model was trained on raw targets (mbt), so predictions are already in raw scale.
    # No inverse transform needed for the target variable.
    
    # Get Actuals
    actuals = np.concatenate([y for x, y in test_ds], axis=0)
    
    # Extract Median (q=0.5) for MAE calculation
    # Predictions shape is (N, 3) -> [q0.1, q0.5, q0.9]
    median_predictions = predictions[:, 1]
    
    mae = float(mean_absolute_error(actuals, median_predictions))
    print(f"GRU Test MAE (Median): {mae}")
    return mae, actuals, predictions

def get_test_dates(input_csv, sequence_length=160):
    print(f"Loading dates from {input_csv}...")
    df = pd.read_csv(input_csv)
    # Assuming first column is date
    dates = pd.to_datetime(df.iloc[:, 0])
    
    n = len(df)
    num_train_samples = int(0.6 * n)
    num_val_samples = int(0.2 * n)
    
    # Test set starts after train + val
    # And targets are shifted by sequence_length
    start_index = num_train_samples + num_val_samples + sequence_length
    
    test_dates = dates.iloc[start_index:].values
    return test_dates

def plot_loss(model_dir, output_path):
    history_path = os.path.join(model_dir, "history.json")
    if not os.path.exists(history_path):
        print(f"No history found at {history_path}, skipping plot.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("<html><body><h1>No training history found</h1></body></html>")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    # Create figure with two subplots
    plt.figure(figsize=(16, 6))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history['loss']) + 1)
    if 'loss' in history:
        plt.plot(epochs, history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: MAE
    plt.subplot(1, 2, 2)
    if 'mae' in history:
        plt.plot(epochs, history['mae'], label='Training MAE')
    if 'val_mae' in history:
        plt.plot(epochs, history['val_mae'], label='Validation MAE')
        
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    html_content = f"""
    <html>
    <head><title>Training History</title></head>
    <body>
        <h1>Training History</h1>
        <img src="data:image/png;base64,{img_base64}" alt="Loss Plot" style="max-width: 100%; border: 1px solid #ddd;">
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"Loss plot saved to {output_path}")

def plot_evaluation_report(actuals, predictions, output_path, metrics_dict=None, dates=None):
    # Ensure 1D arrays for actuals
    actuals = actuals.flatten()
    
    # Predictions are (N, 3) -> [q0.1, q0.5, q0.9]
    q10 = predictions[:, 0]
    q50 = predictions[:, 1]
    q90 = predictions[:, 2]

    # Create figure with GridSpec layout
    # Top: Time Series (Full Width)
    # Bottom: Density (Left), Residuals (Right)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)
    
    # --- Plot 1: Time Series (Last 200 samples) ---
    ax1 = fig.add_subplot(gs[0, :])
    limit = 200
    
    if dates is not None:
        # Ensure dates align with actuals/predictions
        # If dates are longer than actuals (due to batching dropping last partial batch), trim dates
        if len(dates) > len(actuals):
            dates = dates[:len(actuals)]
        elif len(dates) < len(actuals):
            # This shouldn't happen if logic is correct, but handle it
            actuals = actuals[:len(dates)]
            q10 = q10[:len(dates)]
            q50 = q50[:len(dates)]
            q90 = q90[:len(dates)]
            
        plot_dates = dates
        
        # Use tail
        plot_dates_subset = plot_dates[-limit:]
        actuals_subset = actuals[-limit:]
        q10_subset = q10[-limit:]
        q50_subset = q50[-limit:]
        q90_subset = q90[-limit:]
        
        ax1.plot(plot_dates_subset, actuals_subset, label='Actual', color='black', linewidth=1.5, alpha=1.0)
        ax1.plot(plot_dates_subset, q50_subset, label='Median Prediction', color='blue', linewidth=1.5, alpha=0.7, linestyle='--')
        ax1.fill_between(plot_dates_subset, q10_subset, q90_subset, color='blue', alpha=0.2, label='10th-90th Percentile')
        
        date_form = DateFormatter("%Y-%m-%d %H:%M")
        ax1.xaxis.set_major_formatter(date_form)
        ax1.set_xlabel('Time')
    else:
        actuals_subset = actuals[-limit:]
        q10_subset = q10[-limit:]
        q50_subset = q50[-limit:]
        q90_subset = q90[-limit:]
        
        ax1.plot(actuals_subset, label='Actual', color='black', linewidth=1.5, alpha=1.0)
        ax1.plot(q50_subset, label='Median Prediction', color='blue', linewidth=1.5, alpha=0.7, linestyle='--')
        ax1.fill_between(range(len(actuals_subset)), q10_subset, q90_subset, color='blue', alpha=0.2, label='10th-90th Percentile')
        ax1.set_xlabel('Sample Index')

    ax1.set_title(f'Probabilistic Forecast Evaluation (Last {limit} samples)')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Distribution of Actual vs Predicted (Median) ---
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Kernel Density Estimation
    try:
        density_actual = gaussian_kde(actuals)
        density_pred = gaussian_kde(q50)
        
        min_val = min(actuals.min(), q50.min())
        max_val = max(actuals.max(), q50.max())
        padding = (max_val - min_val) * 0.1
        xs = np.linspace(min_val - padding, max_val + padding, 200)
        
        ax2.plot(xs, density_actual(xs), color='green', label='Actual')
        ax2.fill_between(xs, density_actual(xs), color='green', alpha=0.3)
        
        ax2.plot(xs, density_pred(xs), color='orange', label='Predicted (Median)')
        ax2.fill_between(xs, density_pred(xs), color='orange', alpha=0.3)
    except Exception as e:
        print(f"Could not plot KDE: {e}")
        ax2.hist(actuals, bins=30, density=True, alpha=0.5, color='green', label='Actual')
        ax2.hist(q50, bins=30, density=True, alpha=0.5, color='orange', label='Predicted (Median)')
    
    ax2.set_title('Distribution of Actual vs Predicted (Median) Values')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Residuals (using Median) ---
    ax3 = fig.add_subplot(gs[1, 1])
    residuals = actuals - q50
    
    # Histogram
    counts, bins, patches = ax3.hist(residuals, bins=30, density=False, color='purple', alpha=0.6, edgecolor='black')
    
    # KDE for residuals
    try:
        density_res = gaussian_kde(residuals)
        min_res = residuals.min()
        max_res = residuals.max()
        padding_res = (max_res - min_res) * 0.1
        xs_res = np.linspace(min_res - padding_res, max_res + padding_res, 200)
        
        curve = density_res(xs_res)
        bin_width = bins[1] - bins[0]
        scale_factor = len(residuals) * bin_width
        
        ax3.plot(xs_res, curve * scale_factor, color='purple', linewidth=2)
    except:
        pass
    
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax3.set_title('Distribution of Prediction Errors (Residuals from Median)')
    ax3.set_xlabel('Error (Actual - Median Predicted)')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)

    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    metrics_html = ""
    if metrics_dict:
        rows = ""
        for k, v in metrics_dict.items():
            rows += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        metrics_html = f"""
        <div style="margin-bottom: 20px;">
            <h3>Metrics</h3>
            <table border="1" style="border-collapse: collapse; width: 300px;">
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 8px; text-align: left;">Metric</th>
                    <th style="padding: 8px; text-align: left;">Value</th>
                </tr>
                {rows}
            </table>
        </div>
        """

    html_content = f"""
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border: 1px solid #ddd; }}
            th, td {{ text-align: left; padding: 8px; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        {metrics_html}
        <img src="data:image/png;base64,{img_base64}" alt="Evaluation Plots" style="max-width: 100%; border: 1px solid #ddd;">
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"Evaluation report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--input_csv', type=str, required=False)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    parser.add_argument('--plot_output_path', type=str, required=False)
    parser.add_argument('--prediction_plot_path', type=str, required=False)
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading test dataset from {args.test_dataset_path}...")
    test_ds = tf.data.Dataset.load(args.test_dataset_path)
    
    # 2. Evaluate
    mae, actuals, predictions = evaluate_gru(args.model_dir, test_ds)
    
    # 3. Plot Loss
    if args.plot_output_path:
        plot_loss(args.model_dir, args.plot_output_path)
        
    # 4. Plot Evaluation Report (Time Series + Residuals)
    if args.prediction_plot_path:
        dates = None
        if args.input_csv:
            try:
                dates = get_test_dates(args.input_csv)
            except Exception as e:
                print(f"Warning: Could not load dates from CSV: {e}")
        
        metrics_dict = {"MAE": mae}
        plot_evaluation_report(actuals, predictions, args.prediction_plot_path, metrics_dict, dates)
    
    # 5. Save Metrics for Vertex AI
    metrics = {
        "metrics": [
            {
                "name": "mae",
                "numberValue": mae,
                "format": "RAW"
            }
        ]
    }
    
    os.makedirs(os.path.dirname(args.metrics_output_path), exist_ok=True)
    with open(args.metrics_output_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Metrics saved to {args.metrics_output_path}")
