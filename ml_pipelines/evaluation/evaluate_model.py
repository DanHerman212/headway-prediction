"""
Independent Test Set Evaluation

Comprehensive evaluation of trained model on held-out test data with:
- Time series prediction plots (peak vs off-peak periods)
- Route-wise performance analysis
- Error distribution analysis
- Summary reports
"""

import os
import json
from typing import Dict, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from ml_pipelines.config.model_config import ModelConfig
from ml_pipelines.evaluation.metrics import MAESeconds


class ModelEvaluator:
    """
    Evaluate trained model on independent test set with time-series specific analysis.
    
    Usage:
        evaluator = ModelEvaluator(
            model_path="models/headway_model-20260123-120000",
            config=ModelConfig.from_env(),
            output_dir="evaluation_results"
        )
        
        evaluator.load_and_evaluate()
        evaluator.plot_all()
        evaluator.save_summary()
    """
    
    def __init__(
        self,
        model_path: str,
        config: ModelConfig,
        output_dir: str = "evaluation_results"
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to saved model (.keras file)
            config: ModelConfig instance
            output_dir: Directory for saving outputs
        """
        self.model_path = model_path
        self.config = config
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.model = None
        self.test_data = None
        self.predictions = None
        self.metrics = {}
        
        # Route mapping
        self.route_names = ['A', 'C', 'E']
        
    def load_model(self):
        """Load trained model from checkpoint."""
        # Handle case where model_path is a directory containing model.h5
        path_to_load = self.model_path
        if os.path.isdir(path_to_load):
            h5_path = os.path.join(path_to_load, 'model.h5')
            if os.path.exists(h5_path):
                path_to_load = h5_path
                print(f"Found model.h5 in directory, loading from: {path_to_load}")
            else:
                # Check for .keras
                keras_path = os.path.join(path_to_load, 'model.keras')
                if os.path.exists(keras_path):
                    path_to_load = keras_path
                    print(f"Found model.keras in directory, loading from: {path_to_load}")
        
        print(f"Loading model from: {path_to_load}")
        self.model = keras.models.load_model(
            path_to_load,
            custom_objects={'MAESeconds': MAESeconds}
        )
        
    def load_test_data(self, data_path: str = 'data/X.csv'):
        """
        Load and prepare test dataset.
        
        Args:
            data_path: Path to full dataset CSV
        """
        # Load data
        data = pd.read_csv(data_path)
        
        input_x = data.values
        input_t = data['log_headway'].values
        input_r = data[['route_A', 'route_C', 'route_E']].values
        
        # Calculate test split
        n = len(data) - self.config.lookback_steps
        train_end = int(n * self.config.train_split)
        val_end = int(n * (self.config.train_split + self.config.val_split))
        
        # Extract test indices
        test_start = val_end
        test_end = n
        
        # Build test dataset
        def build_test_dataset(start_idx: int, end_idx: int):
            inputs = []
            targets_headway = []
            targets_route = []
            
            for i in range(start_idx, end_idx):
                # Get window
                window = input_x[i:i + self.config.lookback_steps]
                
                # Get targets (at end of window)
                target_idx = i + self.config.lookback_steps
                target_headway = input_t[target_idx]
                target_route = np.argmax(input_r[target_idx])
                
                inputs.append(window)
                targets_headway.append(target_headway)
                targets_route.append(target_route)
            
            X = np.array(inputs, dtype=np.float32)
            y_headway = np.array(targets_headway, dtype=np.float32)
            y_route = np.array(targets_route, dtype=np.int32)
            
            return X, y_headway, y_route
        
        X_test, y_headway_test, y_route_test = build_test_dataset(test_start, test_end)
        
        # Store test data with metadata (extract from last timestep of each window)
        self.test_data = {
            'X': X_test,
            'y_headway': y_headway_test,
            'y_route': y_route_test,
            'hour': X_test[:, -1, data.columns.get_loc('hour')],
            'day_of_week': X_test[:, -1, data.columns.get_loc('day_of_week')],
            'is_weekend': X_test[:, -1, data.columns.get_loc('is_weekend')],
            'route_features': X_test[:, -1, [
                data.columns.get_loc('route_A'),
                data.columns.get_loc('route_C'),
                data.columns.get_loc('route_E')
            ]]
        }
        
        # Create TF dataset for evaluation
        self.test_dataset = tf.data.Dataset.from_tensor_slices((
            X_test,
            {'headway': y_headway_test, 'route': y_route_test}
        )).batch(self.config.batch_size)

    def load_pre_split_test_data(self, data_path: str):
        """
        Load a test dataset that has already been split (contains only test data).
        
        Args:
            data_path: Path to test dataset CSV or Directory
        """
        print(f"DEBUG: Received data_path: {data_path}")
        
        # Standardize target path selection
        # We expect KFP to pass a directory, where we wrote 'test.csv'
        files_to_check = []
        
        if data_path.endswith('.csv'):
            files_to_check.append(data_path)
        else:
            # Check standard names
            files_to_check.append(os.path.join(data_path, 'test.csv'))
            files_to_check.append(os.path.join(data_path, 'test_data.csv')) # legacy compat
        
        data = None
        for path in files_to_check:
            print(f"DEBUG: Checking {path}...")
            if os.path.exists(path):
                try:
                    data = pd.read_csv(path)
                    print(f"SUCCESS: Loaded {len(data)} rows from {path}")
                    break
                except Exception as e:
                    print(f"WARNING: Found file at {path} but failed to read: {e}")
        
        if data is None:
            print(f"CRITICAL: Test data not found in {data_path}")
            print(f"DEBUG: Directory contents of {data_path}:")
            try:
                if os.path.isdir(data_path):
                    print(os.listdir(data_path))
                else:
                    print("(Not a directory)")
            except Exception:
                print("(Could not list directory)")
            
            raise FileNotFoundError(f"Missing test data in {data_path}")
        
        input_x = data.values
        input_t = data['log_headway'].values
        input_r = data[['route_A', 'route_C', 'route_E']].values
        
        # Since this is already the test set, we use the whole thing 
        # (minus lookback for the first sequence)
        n = len(data)
        
        # Build test dataset
        def build_test_dataset():
            inputs = []
            targets_headway = []
            targets_route = []
            
            # We can start from index 0 if valid, but we need lookback_steps history.
            # If the CSV rows are time-sorted, row `i` contains features for time `t`.
            # To predict `t + lookback`, we need rows `i` to `i + lookback`.
            # So we can iterate up to n - lookback.
            
            for i in range(n - self.config.lookback_steps):
                # Get window
                window = input_x[i:i + self.config.lookback_steps]
                
                # Get targets (at end of window)
                target_idx = i + self.config.lookback_steps
                target_headway = input_t[target_idx]
                target_route = np.argmax(input_r[target_idx])
                
                inputs.append(window)
                targets_headway.append(target_headway)
                targets_route.append(target_route)
            
            X = np.array(inputs, dtype=np.float32)
            y_headway = np.array(targets_headway, dtype=np.float32)
            y_route = np.array(targets_route, dtype=np.int32)
            
            return X, y_headway, y_route
        
        X_test, y_headway_test, y_route_test = build_test_dataset()
        
        # Store test data with metadata
        self.test_data = {
            'X': X_test,
            'y_headway': y_headway_test,
            'y_route': y_route_test,
            'hour': X_test[:, -1, data.columns.get_loc('hour')],
            'day_of_week': X_test[:, -1, data.columns.get_loc('day_of_week')],
            'is_weekend': X_test[:, -1, data.columns.get_loc('is_weekend')],
            'route_features': X_test[:, -1, [
                data.columns.get_loc('route_A'),
                data.columns.get_loc('route_C'),
                data.columns.get_loc('route_E')
            ]]
        }
        
        # Create TF dataset for evaluation
        self.test_dataset = tf.data.Dataset.from_tensor_slices((
            X_test,
            {'headway': y_headway_test, 'route': y_route_test}
        )).batch(self.config.batch_size)
    
    def predict(self):
        """Generate predictions on test set."""
        predictions = self.model.predict(self.test_data['X'], batch_size=self.config.batch_size)
        
        # predictions comes back as a list [headway_output, route_output] because
        # we defined the model with outputs=[out_headway, out_route] without specific dictionary output keys
        # unless the model was compiled with output_names that handle that mapping automatically
        # which isn't guaranteed when loading from H5.
        
        if isinstance(predictions, list):
            pred_headway_log = predictions[0].flatten()
            pred_route_probs = predictions[1]
        elif isinstance(predictions, dict):
             # Just in case it DOES return a dict
            pred_headway_log = predictions['headway'].flatten()
            pred_route_probs = predictions['route']
        else:
             # Assume single output if not list/dict? Unexpected for this model.
             raise ValueError(f"Unexpected predictions format: {type(predictions)}")
        
        pred_route = np.argmax(pred_route_probs, axis=1)
        
        # Convert headway from log-space to seconds
        pred_headway_seconds = np.exp(pred_headway_log)
        true_headway_seconds = np.exp(self.test_data['y_headway'])
        
        self.predictions = {
            'headway_log': pred_headway_log,
            'headway_seconds': pred_headway_seconds,
            'route': pred_route,
            'route_probs': pred_route_probs,
            'true_headway_log': self.test_data['y_headway'],
            'true_headway_seconds': true_headway_seconds,
            'true_route': self.test_data['y_route']
        }
        
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics."""
        # Overall metrics
        test_results = self.model.evaluate(self.test_dataset, verbose=0, return_dict=True)
        self.metrics['overall'] = test_results
        
        # Headway metrics in seconds
        mae_seconds = np.mean(np.abs(
            self.predictions['true_headway_seconds'] - self.predictions['headway_seconds']
        ))
        rmse_seconds = np.sqrt(np.mean(
            (self.predictions['true_headway_seconds'] - self.predictions['headway_seconds']) ** 2
        ))
        
        self.metrics['headway_seconds'] = {
            'mae': float(mae_seconds),
            'rmse': float(rmse_seconds)
        }
        
        # Route classification metrics
        route_accuracy = np.mean(
            self.predictions['true_route'] == self.predictions['route']
        )
        self.metrics['route_classification'] = {
            'accuracy': float(route_accuracy)
        }
        
        # Per-route metrics
        self.metrics['per_route'] = {}
        for route_idx, route_name in enumerate(self.route_names):
            mask = self.predictions['true_route'] == route_idx
            if mask.sum() > 0:
                mae = np.mean(np.abs(
                    self.predictions['true_headway_seconds'][mask] - 
                    self.predictions['headway_seconds'][mask]
                ))
                rmse = np.sqrt(np.mean(
                    (self.predictions['true_headway_seconds'][mask] - 
                     self.predictions['headway_seconds'][mask]) ** 2
                ))
                self.metrics['per_route'][route_name] = {
                    'mae_seconds': float(mae),
                    'rmse_seconds': float(rmse),
                    'n_samples': int(mask.sum())
                }
        
        # Peak vs off-peak (rush hour: 7-9am, 5-7pm)
        hour = self.test_data['hour']
        peak_mask = ((hour >= 7) & (hour < 9)) | ((hour >= 17) & (hour < 19))
        
        for period, mask in [('peak', peak_mask), ('off_peak', ~peak_mask)]:
            if mask.sum() > 0:
                mae = np.mean(np.abs(
                    self.predictions['true_headway_seconds'][mask] - 
                    self.predictions['headway_seconds'][mask]
                ))
                self.metrics[f'{period}_hours'] = {
                    'mae_seconds': float(mae),
                    'n_samples': int(mask.sum())
                }
        
    def plot_time_series_windows(self):
        """Plot predictions over multiple time windows."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Headway Predictions: Time Series Windows', fontsize=16, fontweight='bold')
        
        # Define interesting windows
        windows = [
            ('Monday Morning Rush', (0, 7, 9)),  # (day_of_week, hour_start, hour_end)
            ('Wednesday Midday', (2, 11, 14)),
            ('Friday Evening Rush', (4, 17, 20)),
            ('Weekend', (5, 10, 16))
        ]
        
        for ax, (title, (dow, h_start, h_end)) in zip(axes.flatten(), windows):
            # Find samples matching this window
            mask = (
                (self.test_data['day_of_week'] == dow) &
                (self.test_data['hour'] >= h_start) &
                (self.test_data['hour'] < h_end)
            )
            
            if mask.sum() > 0:
                indices = np.where(mask)[0][:200]  # Limit to 200 points for clarity
                
                x_axis = np.arange(len(indices))
                true_vals = self.predictions['true_headway_seconds'][indices]
                pred_vals = self.predictions['headway_seconds'][indices]
                
                ax.plot(x_axis, true_vals, 'o-', label='Actual', alpha=0.7, markersize=3)
                ax.plot(x_axis, pred_vals, 's-', label='Predicted', alpha=0.7, markersize=3)
                ax.fill_between(x_axis, true_vals, pred_vals, alpha=0.2)
                
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Headway (seconds)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                mae = np.mean(np.abs(true_vals - pred_vals))
                ax.text(0.02, 0.98, f'MAE: {mae:.1f}s', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/time_series_windows.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_route_performance(self):
        """Compare performance across routes."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Route-wise Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. MAE comparison
        route_maes = [self.metrics['per_route'][r]['mae_seconds'] for r in self.route_names]
        axes[0].bar(self.route_names, route_maes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0].set_title('MAE by Route', fontweight='bold')
        axes[0].set_ylabel('MAE (seconds)')
        axes[0].set_xlabel('Route')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 2. Error distributions by route
        for route_idx, route_name in enumerate(self.route_names):
            mask = self.predictions['true_route'] == route_idx
            errors = (self.predictions['headway_seconds'][mask] - 
                     self.predictions['true_headway_seconds'][mask])
            axes[1].violinplot([errors], positions=[route_idx], showmeans=True)
        
        axes[1].set_xticks(range(len(self.route_names)))
        axes[1].set_xticklabels(self.route_names)
        axes[1].set_title('Error Distribution by Route', fontweight='bold')
        axes[1].set_ylabel('Prediction Error (seconds)')
        axes[1].set_xlabel('Route')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 3. Route classification confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.predictions['true_route'], self.predictions['route'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                   xticklabels=self.route_names, yticklabels=self.route_names)
        axes[2].set_title('Route Classification Confusion Matrix', fontweight='bold')
        axes[2].set_ylabel('True Route')
        axes[2].set_xlabel('Predicted Route')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/route_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_error_analysis(self):
        """Analyze prediction errors across time dimensions."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Error Analysis', fontsize=16, fontweight='bold')
        
        errors = self.predictions['headway_seconds'] - self.predictions['true_headway_seconds']
        
        # 1. Error vs time of day
        axes[0, 0].scatter(self.test_data['hour'], errors, alpha=0.3, s=10)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Error vs Time of Day', fontweight='bold')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Prediction Error (seconds)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add peak hour shading
        for start, end in [(7, 9), (17, 19)]:
            axes[0, 0].axvspan(start, end, alpha=0.1, color='red', label='Peak Hours' if start == 7 else '')
        axes[0, 0].legend()
        
        # 2. Error distribution by day of week
        error_by_dow = [errors[self.test_data['day_of_week'] == i] for i in range(7)]
        axes[0, 1].boxplot(error_by_dow, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Error Distribution by Day of Week', fontweight='bold')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Prediction Error (seconds)')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Prediction vs actual scatter
        axes[1, 0].scatter(self.predictions['true_headway_seconds'], 
                          self.predictions['headway_seconds'], alpha=0.3, s=10)
        
        # Perfect prediction line
        min_val = min(self.predictions['true_headway_seconds'].min(), 
                     self.predictions['headway_seconds'].min())
        max_val = max(self.predictions['true_headway_seconds'].max(), 
                     self.predictions['headway_seconds'].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Prediction')
        
        axes[1, 0].set_title('Predicted vs Actual Headway', fontweight='bold')
        axes[1, 0].set_xlabel('Actual Headway (seconds)')
        axes[1, 0].set_ylabel('Predicted Headway (seconds)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Error histogram
        axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Prediction Error Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Prediction Error (seconds)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        axes[1, 1].text(0.02, 0.98, 
                       f'Mean: {errors.mean():.2f}s\nStd: {errors.std():.2f}s',
                       transform=axes[1, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_peak_comparison(self):
        """Compare predictions during peak vs off-peak hours."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Peak vs Off-Peak Performance', fontsize=16, fontweight='bold')
        
        hour = self.test_data['hour']
        peak_mask = ((hour >= 7) & (hour < 9)) | ((hour >= 17) & (hour < 19))
        
        for ax, (title, mask) in zip(axes, [('Peak Hours', peak_mask), ('Off-Peak Hours', ~peak_mask)]):
            indices = np.where(mask)[0][:300]  # Limit for clarity
            
            x_axis = np.arange(len(indices))
            true_vals = self.predictions['true_headway_seconds'][indices]
            pred_vals = self.predictions['headway_seconds'][indices]
            
            ax.plot(x_axis, true_vals, 'o-', label='Actual', alpha=0.6, markersize=2)
            ax.plot(x_axis, pred_vals, 's-', label='Predicted', alpha=0.6, markersize=2)
            
            mae = np.mean(np.abs(true_vals - pred_vals))
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Headway (seconds)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.text(0.02, 0.98, f'MAE: {mae:.1f}s\nSamples: {mask.sum()}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/peak_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_summary(self):
        """Save evaluation summary to JSON and text."""
        # Save metrics as JSON
        metrics_path = f'{self.output_dir}/test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Create text summary
        summary_path = f'{self.output_dir}/evaluation_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TEST SET EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Test samples: {len(self.test_data['y_headway'])}\n\n")
            
            f.write("OVERALL METRICS\n")
            f.write("-" * 70 + "\n")
            for key, value in self.metrics['overall'].items():
                f.write(f"{key}: {value:.4f}\n")
            
            f.write(f"\nHEADWAY PREDICTION (in seconds)\n")
            f.write("-" * 70 + "\n")
            f.write(f"MAE: {self.metrics['headway_seconds']['mae']:.2f} seconds\n")
            f.write(f"RMSE: {self.metrics['headway_seconds']['rmse']:.2f} seconds\n")
            
            f.write(f"\nROUTE CLASSIFICATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy: {self.metrics['route_classification']['accuracy']:.4f}\n")
            
            f.write(f"\nPER-ROUTE PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            for route, metrics in self.metrics['per_route'].items():
                f.write(f"\nRoute {route}:\n")
                f.write(f"  MAE: {metrics['mae_seconds']:.2f} seconds\n")
                f.write(f"  RMSE: {metrics['rmse_seconds']:.2f} seconds\n")
                f.write(f"  Samples: {metrics['n_samples']}\n")
            
            f.write(f"\nPEAK VS OFF-PEAK\n")
            f.write("-" * 70 + "\n")
            for period in ['peak_hours', 'off_peak_hours']:
                if period in self.metrics:
                    f.write(f"{period.replace('_', ' ').title()}:\n")
                    f.write(f"  MAE: {self.metrics[period]['mae_seconds']:.2f} seconds\n")
                    f.write(f"  Samples: {self.metrics[period]['n_samples']}\n")
        
        print(f"Summary saved to {summary_path}")
        
    def load_and_evaluate(self, data_path: str = 'data/X.csv', is_pre_split: bool = False):
        """Run complete evaluation pipeline."""
        print("Loading model...")
        self.load_model()
        
        print(f"Loading test data (pre-split={is_pre_split})...")
        if is_pre_split:
            self.load_pre_split_test_data(data_path)
        else:
            self.load_test_data(data_path)
        
        print("Generating predictions...")
        self.predict()
        
        print("Calculating metrics...")
        self.calculate_metrics()
        
    def plot_all(self):
        """Generate all evaluation plots."""
        print("Plotting time series windows...")
        self.plot_time_series_windows()
        
        print("Plotting route performance...")
        self.plot_route_performance()
        
        print("Plotting error analysis...")
        self.plot_error_analysis()
        
        print("Plotting peak comparison...")
        self.plot_peak_comparison()
        
        print(f"All plots saved to {self.output_dir}/")


def main():
    """
    Run evaluationml_pipelines.evaluation.evaluate_model --model ... --data ...
    """
    import argparse
    import sys
    
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Now we can invoke the class which imports from ml_pipelines or config
    # But wait, the class uses 'from evaluation.metrics'.
    # If we run as module 'ml_pipelines.evaluation.evaluate_model', imports are relative? 
    # Or absolute. 'from evaluation.metrics' implies evaluation is top level.
    # We should probably fix the imports in this file to be robust, 
    # but for now let's ensure 'ml_pipelines' directory is in path (which the above does NOT do, it adds parent).
    # If we add 'project_root/ml_pipelines' to path:
    sys.path.append(os.path.join(project_root, 'ml_pipelines'))
    
    
    parser = argparse.ArgumentParser(description='Evaluate trained headway prediction model')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data', type=str, default='data/X.csv', help='Path to data CSV (full or pre-split)')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--pre_split', action='store_true', help='Treat data as pre-split test set')
    # Argument for pipeline metrics output (optional, but good for KFP)
    parser.add_argument('--metrics_output', type=str, default=None, help='Path to save KFP metrics JSON')
    
    args = parser.parse_args()
    
    # Load config
    config = ModelConfig.from_env()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        config=config,
        output_dir=args.output
    )
    
    # Run evaluation
    evaluator.load_and_evaluate(data_path=args.data, is_pre_split=args.pre_split)
    evaluator.plot_all()
    evaluator.save_summary()
    
    # Save KFP metrics if requested
    if args.metrics_output:
        import json
        # We need to extract simple key-value pairs for KFP
        kfp_metrics = {
            'mae_seconds': evaluator.metrics['headway_seconds']['mae'],
            'rmse_seconds': evaluator.metrics['headway_seconds']['rmse'],
            'route_accuracy': evaluator.metrics['route_classification']['accuracy']
        }
        with open(args.metrics_output, 'w') as f:
            json.dump(kfp_metrics, f)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {args.output}/")
    print(f"  - test_metrics.json")
    print(f"  - evaluation_summary.txt")
    print(f"  - time_series_windows.png")
    print(f"  - route_performance.png")
    print(f"  - error_analysis.png")
    print(f"  - peak_comparison.png")


if __name__ == "__main__":
    main()
