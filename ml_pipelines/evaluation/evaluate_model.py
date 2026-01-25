"""Model evaluation script."""

import logging
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates the trained model on test data."""
    
    def __init__(self, model_path: str, test_data_path: str, output_dir: str):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to the saved Keras model
            test_data_path: Path to the test dataset CSV
            output_dir: Directory to save evaluation artifacts
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.output_dir = output_dir
        self.model = None
        self.test_data = None
        
        # Ensure output directory exists (locally)
        os.makedirs(self.output_dir, exist_ok=True)

    def load_resources(self) -> None:
        """Load the model and test data with robust GCS handling."""
        logger.info(f"Loading model from {self.model_path}")
        
        # 1. Load Model
        try:
            # OPTION 1: Try loading specific .keras file first (preferred for preserving Keras API)
            keras_file_path = os.path.join(self.model_path, "model.keras")
            if os.path.exists(keras_file_path):
                logger.info(f"Found .keras backup file at {keras_file_path}. Loading using Keras format.")
                # compile=False avoids loading the optimizer state, which often causes version conflicts (e.g. Adam.build missing)
                # We don't need the optimizer for evaluation/inference.
                self.model = tf.keras.models.load_model(keras_file_path, compile=False)
            else:
                # Fallback to standard directory load (SavedModel)
                logger.info(f"Loading model from directory {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
        logger.info(f"Loading test data from {self.test_data_path}")
        
        # 2. Load Data (Handle GCS vs Local)
        # Check if the path is a directory (KFP often passes a dir for Dataset artifact)
        # If it is a directory, look for the CSV file inside.
        final_csv_path = self.test_data_path
        
        # Heuristic: If path doesn't end in .csv, check if it's a dir with a csv
        if not self.test_data_path.endswith('.csv'):
             # If using native GCS client, listing blobs is hard without explicit bucket knowledge here.
             # But if it's a local path (or Fuse path), we can check os.path.exists
             if os.path.isdir(self.test_data_path):
                 possible_files = [f for f in os.listdir(self.test_data_path) if f.endswith('.csv')]
                 if possible_files:
                     final_csv_path = os.path.join(self.test_data_path, possible_files[0])
                     logger.info(f"Found CSV in directory: {final_csv_path}")
                 else:
                     # It might be 'test_data.csv' by convention from train.py
                     candidate = os.path.join(self.test_data_path, 'test_data.csv')
                     final_csv_path = candidate
                     logger.info(f"Assuming CSV path: {final_csv_path}")
        
        if final_csv_path.startswith('/gcs/'):
            # GCS Fuse Path - Try Native Client first for consistency
            try:
                from google.cloud import storage
                import io
                
                logger.info("Detected GCS path. Attempting Native Client download.")
                path_parts = final_csv_path[5:].split('/')
                bucket_name = path_parts[0]
                blob_name = '/'.join(path_parts[1:])
                
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                
                content = blob.download_as_string()
                self.test_data = pd.read_csv(io.BytesIO(content))
                logger.info("Successfully loaded data via Native Client.")
            except Exception as e:
                logger.warning(f"Native Client failed ({e}). Falling back to FileSystem path.")
                self.test_data = pd.read_csv(final_csv_path)
        else:
            # Standard Local Path
            self.test_data = pd.read_csv(final_csv_path)
            
        logger.info(f"Test data loaded: {self.test_data.shape[0]} rows.")

    def _recover_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recover readable 'hour' and 'day' from cyclically encoded features.
        
        Args:
            df: DataFrame containing *_sin and *_cos columns
            
        Returns:
            DataFrame with added 'hour' and 'day_name' columns.
        """
        df_copy = df.copy()
        
        # Recover Hour (0-23)
        if 'hour_sin' in df_copy.columns and 'hour_cos' in df_copy.columns:
            # arctan2 returns (-pi, pi]
            # hour_radians was: 2 * pi * hour / 24
            # so: hour = (radians * 24) / (2 * pi)
            radians = np.arctan2(df_copy['hour_sin'], df_copy['hour_cos'])
            # Convert negative radians to positive (0 to 2pi)
            radians = np.where(radians < 0, radians + 2*np.pi, radians)
            hours = (radians * 24) / (2 * np.pi)
            df_copy['hour'] = np.round(hours).astype(int) % 24
            logger.info("Recovered 'hour' column from cyclical features.")
        else:
            logger.warning("Could not find hour_sin/cos to recover hour.")
            df_copy['hour'] = 0 # Fallback
            
        # Recover Day (0-6)
        if 'day_sin' in df_copy.columns and 'day_cos' in df_copy.columns:
            radians = np.arctan2(df_copy['day_sin'], df_copy['day_cos'])
            radians = np.where(radians < 0, radians + 2*np.pi, radians)
            days = (radians * 7) / (2 * np.pi)
            df_copy['day_idx'] = np.round(days).astype(int) % 7
            
            day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            df_copy['day_name'] = df_copy['day_idx'].map(day_map)
            logger.info("Recovered 'day_name' from cyclical features.")
        else:
             df_copy['day_name'] = 'Unknown'

        return df_copy
        
    def evaluate(self, metrics_output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run evaluation: Predict, Calculate Metrics, Generate Plots.
        
        Args:
            metrics_output_path: Optional path to write metrics JSON for KFP.
        """
        if self.model is None or self.test_data is None:
            self.load_resources()
            
        # 1. Prepare Data for Prediction
        # Feature columns must match training (all columns except log_headway)
        # Assuming the CSV structure: [log_headway, route_*, time_features...]
        # We need to drop input_t (target) to get input_x
        target_col = 'log_headway'
        
        if target_col not in self.test_data.columns:
            raise KeyError(f"Target column '{target_col}' not found in test data.")
            
        X = self.test_data.drop(columns=[target_col]).values
        y_true_log = self.test_data[target_col].values
        
        # 2. Run Prediction
        logger.info("Running predictions...")
        # Model returns [headway, routes] (multi-output)
        predictions = self.model.predict(X)
        
        # Check structure: usually [headway_pred, route_pred]
        if isinstance(predictions, list):
            y_pred_log = predictions[0].flatten()  # Headway is first output
        else:
            y_pred_log = predictions.flatten()
            
        # Inverse transform log1p -> expm1
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)
        
        # Clean predictions (no negative times)
        y_pred = np.maximum(y_pred, 0)
        
        # 3. Calculate Metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        metrics = {
            "MAE_seconds": float(mae),
            "RMSE_seconds": float(rmse),
            "Test_Size": int(len(y_true))
        }
        logger.info(f"Metrics: {metrics}")
        
        # 4. Generate Plots
        # Recover metadata for plotting
        plotting_df = self._recover_time_features(self.test_data)
        plotting_df['Actual'] = y_true
        plotting_df['Predicted'] = y_pred
        
        self._plot_actual_vs_predicted(y_true, y_pred)
        self._plot_error_distribution(y_true, y_pred)
        self._plot_metric_by_hour(plotting_df)
        
        # Save metrics JSON (Local Dir)
        local_metrics_path = os.path.join(self.output_dir, 'metrics.json')
        with open(local_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Save metrics JSON (KFP Output)
        if metrics_output_path:
            # Ensure dir exists
            os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
            with open(metrics_output_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics written to KFP output: {metrics_output_path}")
            
        return metrics

    def _plot_actual_vs_predicted(self, y_true, y_pred):
        """Scatter plot."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Headway (s)')
        plt.ylabel('Predicted Headway (s)')
        plt.title('Actual vs Predicted Headway')
        plt.savefig(os.path.join(self.output_dir, 'actual_vs_predicted.png'))
        plt.close()

    def _plot_error_distribution(self, y_true, y_pred):
        """Error histogram."""
        errors = y_pred - y_true
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black')
        plt.xlabel('Prediction Error (s)')
        plt.ylabel('Count')
        plt.title('Prediction Error Distribution')
        plt.savefig(os.path.join(self.output_dir, 'error_distribution.png'))
        plt.close()
        
    def _plot_metric_by_hour(self, df):
        """MAE by Hour."""
        if 'hour' not in df.columns:
            logger.warning("Skipping hourly plot: 'hour' column missing.")
            return

        df['abs_error'] = np.abs(df['Predicted'] - df['Actual'])
        hourly_mae = df.groupby('hour')['abs_error'].mean()
        
        plt.figure(figsize=(12, 6))
        hourly_mae.plot(kind='bar')
        plt.xlabel('Hour of Day')
        plt.ylabel('MAE (s)')
        plt.title('Mean Absolute Error by Hour')
        plt.grid(axis='y')
        plt.savefig(os.path.join(self.output_dir, 'mae_by_hour.png'))
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # Updated arguments to match pipeline definition
    parser.add_argument('--model', type=str, required=True, help="Path to model directory or file")
    parser.add_argument('--data', type=str, required=True, help="Path to test data CSV or directory")
    parser.add_argument('--output', type=str, required=True, help="Path to output plots directory")
    parser.add_argument('--metrics_output', type=str, required=False, help="Path to write metrics JSON")
    
    # Legacy/Optional flags from pipeline
    parser.add_argument('--pre_split', action='store_true', help="Flag indicating data is pre-split")

    args = parser.parse_args()
    
    try:
        evaluator = ModelEvaluator(
            model_path=args.model,
            test_data_path=args.data,
            output_dir=args.output
        )
        evaluator.evaluate(metrics_output_path=args.metrics_output)
        logger.info("Evaluation complete.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import sys
        sys.exit(1)
