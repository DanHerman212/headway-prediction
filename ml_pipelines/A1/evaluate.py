"""
Evaluation Module for A1 Track Model

Evaluates trained model on independent test set with:
- Classification metrics: F1 score for route prediction
- Regression metrics: MAE in seconds for headway prediction
- Visualizations: confusion matrix, prediction vs actual, residuals, time series

Usage:
    python evaluate.py --run_name exp01-baseline
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from config import config
from train import load_preprocessed_data, calculate_split_indices, create_timeseries_datasets


def load_model_and_data(run_name: str) -> Tuple[keras.Model, tf.data.Dataset, np.ndarray, Dict]:
    """
    Load trained model and test data.
    
    Args:
        run_name: Training run identifier
    
    Returns:
        Tuple of (model, test_dataset, full_data_array, metadata)
    """
    print(f"Loading model from run: {run_name}")
    
    # Load model
    model_path = config.checkpoint_path
    print(f"  Model: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = keras.models.load_model(model_path)
    print(f"  ✓ Model loaded")
    
    # Load data
    X, metadata = load_preprocessed_data()
    
    # Create datasets
    train_end, val_end, test_end = calculate_split_indices(X.shape[0])
    _, _, test_ds = create_timeseries_datasets(X, train_end, val_end, test_end)
    
    print(f"  ✓ Test data loaded")
    
    return model, test_ds, X, metadata


def inverse_transform_headway(log_headway: np.ndarray) -> np.ndarray:
    """
    Convert log-scaled headway back to minutes.
    
    Formula: exp(log_headway) - offset
    
    Args:
        log_headway: Log-transformed headway values
    
    Returns:
        Headway in minutes
    """
    return np.exp(log_headway) - config.LOG_OFFSET


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str) -> Dict:
    """
    Evaluate route classification performance.
    
    Args:
        y_true: True route labels (one-hot)
        y_pred: Predicted route probabilities (one-hot)
        output_dir: Directory to save plots
    
    Returns:
        Dictionary of classification metrics
    """
    print("\n" + "="*60)
    print("Classification Metrics (Route Prediction)")
    print("="*60)
    
    # Convert one-hot to class indices
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Class names
    class_names = ['Route A', 'Route C', 'Route E']
    
    # Classification report
    report = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=class_names,
        digits=4
    )
    print("\nClassification Report:")
    print(report)
    
    # F1 scores
    f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro')
    f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted')
    
    print(f"\nF1 Scores:")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Route Classification Confusion Matrix')
    plt.ylabel('True Route')
    plt.xlabel('Predicted Route')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"\n✓ Confusion matrix saved: {cm_path}")
    plt.close()
    
    return {
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist()
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str) -> Dict:
    """
    Evaluate headway regression performance.
    
    Args:
        y_true: True log-scaled headway values
        y_pred: Predicted log-scaled headway values
        output_dir: Directory to save plots
    
    Returns:
        Dictionary of regression metrics
    """
    print("\n" + "="*60)
    print("Regression Metrics (Headway Prediction)")
    print("="*60)
    
    # Inverse transform to get real headways in minutes
    y_true_minutes = inverse_transform_headway(y_true.flatten())
    y_pred_minutes = inverse_transform_headway(y_pred.flatten())
    
    # Convert to seconds for metrics
    y_true_seconds = y_true_minutes * 60
    y_pred_seconds = y_pred_minutes * 60
    
    # Calculate metrics
    mae_seconds = np.mean(np.abs(y_true_seconds - y_pred_seconds))
    rmse_seconds = np.sqrt(np.mean((y_true_seconds - y_pred_seconds)**2))
    mape = np.mean(np.abs((y_true_seconds - y_pred_seconds) / y_true_seconds)) * 100
    
    print(f"\nMetrics (in seconds):")
    print(f"  MAE: {mae_seconds:.2f} seconds")
    print(f"  RMSE: {rmse_seconds:.2f} seconds")
    print(f"  MAPE: {mape:.2f}%")
    
    print(f"\nMetrics (in minutes):")
    print(f"  MAE: {mae_seconds/60:.2f} minutes")
    print(f"  RMSE: {rmse_seconds/60:.2f} minutes")
    
    # 1. Scatter plot: Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true_minutes, y_pred_minutes, alpha=0.3, s=10)
    plt.plot([0, y_true_minutes.max()], [0, y_true_minutes.max()], 'r--', label='Perfect prediction')
    plt.xlabel('True Headway (minutes)')
    plt.ylabel('Predicted Headway (minutes)')
    plt.title('Predicted vs Actual Headway')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    pred_vs_actual_path = os.path.join(output_dir, 'predicted_vs_actual.png')
    plt.savefig(pred_vs_actual_path, dpi=150)
    print(f"\n✓ Predicted vs actual plot saved: {pred_vs_actual_path}")
    plt.close()
    
    # 2. Residual plot
    residuals_minutes = y_pred_minutes - y_true_minutes
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true_minutes, residuals_minutes, alpha=0.3, s=10)
    plt.axhline(y=0, color='r', linestyle='--', label='Zero error')
    plt.xlabel('True Headway (minutes)')
    plt.ylabel('Residual (Predicted - True, minutes)')
    plt.title('Residual Plot')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    residual_path = os.path.join(output_dir, 'residuals.png')
    plt.savefig(residual_path, dpi=150)
    print(f"✓ Residual plot saved: {residual_path}")
    plt.close()
    
    # 3. Error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals_minutes, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero error')
    plt.xlabel('Residual (Predicted - True, minutes)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    error_dist_path = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(error_dist_path, dpi=150)
    print(f"✓ Error distribution saved: {error_dist_path}")
    plt.close()
    
    return {
        'mae_seconds': float(mae_seconds),
        'rmse_seconds': float(rmse_seconds),
        'mape': float(mape),
        'mae_minutes': float(mae_seconds / 60),
        'rmse_minutes': float(rmse_seconds / 60)
    }


def plot_time_series_sample(
    y_true_route: np.ndarray,
    y_pred_route: np.ndarray,
    y_true_headway: np.ndarray,
    y_pred_headway: np.ndarray,
    output_dir: str,
    n_samples: int = 200
):
    """
    Plot time series overlay of predictions vs actuals.
    
    Args:
        y_true_route: True route labels
        y_pred_route: Predicted route labels
        y_true_headway: True headway values (log-scaled)
        y_pred_headway: Predicted headway values (log-scaled)
        output_dir: Directory to save plot
        n_samples: Number of samples to plot
    """
    print(f"\nPlotting time series sample (first {n_samples} predictions)...")
    
    # Inverse transform headways
    y_true_minutes = inverse_transform_headway(y_true_headway.flatten())
    y_pred_minutes = inverse_transform_headway(y_pred_headway.flatten())
    
    # Limit to n_samples
    y_true_minutes = y_true_minutes[:n_samples]
    y_pred_minutes = y_pred_minutes[:n_samples]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(n_samples)
    ax.plot(x, y_true_minutes, 'b-', label='True Headway', alpha=0.7, linewidth=1.5)
    ax.plot(x, y_pred_minutes, 'r--', label='Predicted Headway', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Headway (minutes)')
    ax.set_title(f'Time Series: True vs Predicted Headway (First {n_samples} samples)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    ts_path = os.path.join(output_dir, 'time_series_overlay.png')
    plt.savefig(ts_path, dpi=150)
    print(f"✓ Time series overlay saved: {ts_path}")
    plt.close()


def calculate_baseline_metrics(X: np.ndarray, val_end: int, test_end: int) -> Dict:
    """
    Calculate baseline model performance for comparison.
    
    Baseline: Persistence model (next headway = last headway)
    
    Args:
        X: Full preprocessed data array
        val_end: End index of validation set
        test_end: End index of test set
    
    Returns:
        Dictionary of baseline metrics
    """
    print("\n" + "="*60)
    print("Baseline Model (Persistence)")
    print("="*60)
    
    # Extract test set headways (log-scaled)
    test_headways_log = X[val_end:test_end, 0]
    
    # Persistence: predict t+1 = t
    y_true_log = test_headways_log[config.LOOKBACK_WINDOW + 1:]
    y_pred_log = test_headways_log[config.LOOKBACK_WINDOW:-1]
    
    # Inverse transform
    y_true_minutes = inverse_transform_headway(y_true_log)
    y_pred_minutes = inverse_transform_headway(y_pred_log)
    
    # Convert to seconds
    y_true_seconds = y_true_minutes * 60
    y_pred_seconds = y_pred_minutes * 60
    
    # Calculate metrics
    baseline_mae = np.mean(np.abs(y_true_seconds - y_pred_seconds))
    baseline_rmse = np.sqrt(np.mean((y_true_seconds - y_pred_seconds)**2))
    
    print(f"\nPersistence Baseline:")
    print(f"  MAE: {baseline_mae:.2f} seconds ({baseline_mae/60:.2f} minutes)")
    print(f"  RMSE: {baseline_rmse:.2f} seconds ({baseline_rmse/60:.2f} minutes)")
    
    return {
        'baseline_mae_seconds': float(baseline_mae),
        'baseline_rmse_seconds': float(baseline_rmse),
        'baseline_mae_minutes': float(baseline_mae / 60),
        'baseline_rmse_minutes': float(baseline_rmse / 60)
    }


def evaluate_model(run_name: str) -> Dict:
    """
    Complete evaluation pipeline.
    
    Args:
        run_name: Training run identifier
    
    Returns:
        Dictionary with all evaluation metrics
    """
    print("="*80)
    print(f"Evaluating A1 Model - Run: {run_name}")
    print("="*80)
    
    # Create output directory
    output_dir = os.path.join(config.CHECKPOINT_DIR, f"{run_name}_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load model and data
    model, test_ds, X, metadata = load_model_and_data(run_name)
    
    # Get predictions
    print("\nGenerating predictions on test set...")
    predictions = model.predict(test_ds, verbose=1)
    
    # Extract predictions and ground truth
    y_pred_route = predictions['route_output']
    y_pred_headway = predictions['headway_output']
    
    # Collect ground truth from dataset
    y_true_route_list = []
    y_true_headway_list = []
    
    for _, targets in test_ds:
        y_true_route_list.append(targets['route_output'].numpy())
        y_true_headway_list.append(targets['headway_output'].numpy())
    
    y_true_route = np.concatenate(y_true_route_list, axis=0)
    y_true_headway = np.concatenate(y_true_headway_list, axis=0)
    
    # Ensure shapes match (trim if needed due to batching)
    min_len = min(len(y_true_route), len(y_pred_route))
    y_true_route = y_true_route[:min_len]
    y_pred_route = y_pred_route[:min_len]
    y_true_headway = y_true_headway[:min_len]
    y_pred_headway = y_pred_headway[:min_len]
    
    print(f"  Predictions shape: {y_pred_headway.shape}")
    
    # Evaluate classification
    classification_metrics = evaluate_classification(y_true_route, y_pred_route, output_dir)
    
    # Evaluate regression
    regression_metrics = evaluate_regression(y_true_headway, y_pred_headway, output_dir)
    
    # Plot time series
    plot_time_series_sample(
        y_true_route, y_pred_route,
        y_true_headway, y_pred_headway,
        output_dir
    )
    
    # Calculate baseline
    train_end, val_end, test_end = calculate_split_indices(X.shape[0])
    baseline_metrics = calculate_baseline_metrics(X, val_end, test_end)
    
    # Compare model vs baseline
    print("\n" + "="*60)
    print("Model vs Baseline Comparison")
    print("="*60)
    improvement = (baseline_metrics['baseline_mae_seconds'] - regression_metrics['mae_seconds']) / baseline_metrics['baseline_mae_seconds'] * 100
    print(f"\nMAE Improvement over Baseline: {improvement:.2f}%")
    print(f"  Baseline: {baseline_metrics['baseline_mae_seconds']:.2f} seconds")
    print(f"  Model: {regression_metrics['mae_seconds']:.2f} seconds")
    
    # Combine all metrics
    results = {
        'run_name': run_name,
        'classification': classification_metrics,
        'regression': regression_metrics,
        'baseline': baseline_metrics,
        'improvement_pct': float(improvement)
    }
    
    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {results_path}")
    
    return results


def main():
    """Command-line interface for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate A1 track model')
    parser.add_argument(
        '--run_name',
        type=str,
        required=True,
        help='Training run identifier'
    )
    
    args = parser.parse_args()
    
    results = evaluate_model(run_name=args.run_name)
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"\nKey Results:")
    print(f"  Route F1 (macro): {results['classification']['f1_macro']:.4f}")
    print(f"  Headway MAE: {results['regression']['mae_seconds']:.2f} seconds")
    print(f"  Improvement over baseline: {results['improvement_pct']:.2f}%")
    
    output_dir = os.path.join(config.CHECKPOINT_DIR, f"{args.run_name}_evaluation")
    print(f"\nPlots saved in: {output_dir}")


if __name__ == "__main__":
    main()
