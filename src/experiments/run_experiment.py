#!/usr/bin/env python3
"""
Single entry point for running training experiments on Vertex AI.

Integrates with Vertex AI Experiments for full experiment tracking,
including hyperparameters, time-series metrics, and model artifacts.

Usage:
    python -m src.experiments.run_experiment --exp_id 1
    python -m src.experiments.run_experiment --exp_id 2 --data_dir gs://bucket/data
"""

import argparse
import os
import json
from datetime import datetime

import tensorflow as tf
import numpy as np
from google.cloud import aiplatform

from src.experiments.experiment_config import get_experiment, list_experiments, ExperimentConfig
from src.models.st_convnet import HeadwayConvLSTM
from src.data.dataset import SubwayDataGenerator
from src.config import Config
from src.metrics import rmse_seconds, r_squared


# ============================================================================
# Vertex AI Experiments Callback
# ============================================================================

class VertexExperimentCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that logs metrics to Vertex AI Experiments in real-time.
    
    This enables:
    - Live tracking of training progress in Vertex AI Console
    - Comparison of metrics across runs
    - Programmatic querying of results
    """
    
    def __init__(self, log_every_n_epochs: int = 1):
        """
        Args:
            log_every_n_epochs: How often to log metrics (default: every epoch)
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of each epoch."""
        if logs is None:
            return
        
        if (epoch + 1) % self.log_every_n_epochs != 0:
            return
        
        # Log time-series metrics
        metrics_to_log = {}
        
        # Map Keras metric names to cleaner names
        metric_mapping = {
            'loss': 'train_loss',
            'val_loss': 'val_loss',
            'rmse_seconds': 'train_rmse_seconds',
            'val_rmse_seconds': 'val_rmse_seconds',
            'r_squared': 'train_r_squared',
            'val_r_squared': 'val_r_squared',
        }
        
        for keras_name, display_name in metric_mapping.items():
            if keras_name in logs:
                metrics_to_log[display_name] = float(logs[keras_name])
        
        # Also log learning rate if available
        if hasattr(self.model.optimizer, 'learning_rate'):
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                metrics_to_log['learning_rate'] = float(lr.numpy())
            else:
                metrics_to_log['learning_rate'] = float(lr)
        
        # Log to Vertex AI Experiments
        try:
            aiplatform.log_time_series_metrics(metrics_to_log, step=epoch + 1)
        except Exception as e:
            print(f"Warning: Failed to log metrics to Vertex AI: {e}")


# ============================================================================
# Data & Model Utilities
# ============================================================================

def download_gcs_data(gcs_data_dir: str, local_dir: str = "/tmp/data") -> str:
    """
    Download data files from GCS to local directory.
    
    Args:
        gcs_data_dir: GCS path like gs://bucket/path/to/data
        local_dir: Local directory to download to
        
    Returns:
        Local directory path containing the downloaded files
    """
    from google.cloud import storage
    
    os.makedirs(local_dir, exist_ok=True)
    
    # Parse GCS path
    # gs://bucket-name/path/to/data -> bucket-name, path/to/data
    path = gcs_data_dir.replace("gs://", "")
    bucket_name = path.split("/")[0]
    prefix = "/".join(path.split("/")[1:])
    
    print(f"Downloading data from {gcs_data_dir} to {local_dir}...")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Files we need
    required_files = [
        "headway_matrix_full.npy",
        "schedule_matrix_full.npy",
    ]
    
    for filename in required_files:
        blob_path = f"{prefix}/{filename}" if prefix else filename
        blob = bucket.blob(blob_path)
        local_path = os.path.join(local_dir, filename)
        
        print(f"  Downloading {blob_path}...")
        blob.download_to_filename(local_path)
        print(f"  Saved to {local_path}")
    
    print("Download complete!")
    return local_dir


def setup_gcs_auth():
    """Setup GCS authentication if running on Vertex AI."""
    # Vertex AI automatically provides credentials via the environment
    # This function can be extended for local testing with service accounts
    pass


def upload_to_gcs(local_dir: str, gcs_dir: str):
    """
    Upload local directory contents to GCS.
    
    Args:
        local_dir: Local directory path
        gcs_dir: GCS path like gs://bucket/path/to/dir
    """
    from google.cloud import storage
    
    # Parse GCS path
    path = gcs_dir.replace("gs://", "")
    bucket_name = path.split("/")[0]
    prefix = "/".join(path.split("/")[1:])
    
    print(f"Uploading results from {local_dir} to {gcs_dir}...")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            # Calculate relative path from local_dir
            rel_path = os.path.relpath(local_path, local_dir)
            blob_path = f"{prefix}/{rel_path}" if prefix else rel_path
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"  Uploaded: {blob_path}")
    
    print("Upload complete!")


def create_callbacks(config: ExperimentConfig, local_output_dir: str):
    """Create training callbacks with local checkpointing."""
    
    os.makedirs(local_output_dir, exist_ok=True)
    
    # Paths - always use local paths for checkpoints
    checkpoint_path = os.path.join(local_output_dir, "best_model.keras")
    tensorboard_dir = os.path.join(local_output_dir, "tensorboard")
    
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        # Model checkpointing - save locally
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # TensorBoard logging - save locally
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
    ]
    
    return callbacks


def run_experiment(
    exp_id: int, 
    data_dir: str = None, 
    output_dir: str = None,
    project: str = "time-series-478616",
    location: str = "us-east1",
    experiment_name: str = "headway-regularization",
):
    """
    Run a single training experiment with Vertex AI Experiments tracking.
    
    Args:
        exp_id: Experiment ID (1-4)
        data_dir: Override data directory (for GCS paths)
        output_dir: Override output directory (for GCS paths)
        project: GCP project ID
        location: GCP region
        experiment_name: Name of the Vertex AI Experiment to log to
    """
    # Get experiment config
    exp_config = get_experiment(exp_id)
    
    # Override paths if provided
    if data_dir:
        exp_config.data_dir = data_dir
    if output_dir:
        exp_config.output_dir = output_dir
    
    print("=" * 60)
    print(f"Running Experiment {exp_id}: {exp_config.exp_name}")
    print("=" * 60)
    print(f"Description: {exp_config.description}")
    print(f"Data dir: {exp_config.data_dir}")
    print(f"Output dir: {exp_config.experiment_output_dir}")
    print()
    
    # Initialize Vertex AI
    print("Initializing Vertex AI Experiments...")
    aiplatform.init(project=project, location=location, experiment=experiment_name)
    
    # Generate unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"exp{exp_id:02d}-{exp_config.exp_name}-{timestamp}"
    print(f"Using experiment: {experiment_name}")
    print()
    
    # If data is on GCS, download to local /tmp
    actual_data_dir = exp_config.data_dir
    if actual_data_dir.startswith("gs://"):
        actual_data_dir = download_gcs_data(exp_config.data_dir)
    
    # Determine local output directory (save locally, upload to GCS after)
    gcs_output_dir = None
    if exp_config.experiment_output_dir.startswith("gs://"):
        gcs_output_dir = exp_config.experiment_output_dir
        local_output_dir = f"/tmp/exp_{exp_id:02d}_{exp_config.exp_name}"
    else:
        local_output_dir = exp_config.experiment_output_dir
    
    os.makedirs(local_output_dir, exist_ok=True)
    
    print(f"Loading data from {actual_data_dir}...")
    print(f"Local output dir: {local_output_dir}")
    if gcs_output_dir:
        print(f"Will upload to: {gcs_output_dir}")
    print()
    
    # Create base config for data generator
    base_config = Config()
    base_config.DATA_DIR = actual_data_dir  # Use local path (downloaded from GCS if needed)
    base_config.BATCH_SIZE = exp_config.batch_size
    base_config.LOOKBACK_MINS = exp_config.lookback_mins
    base_config.FORECAST_MINS = exp_config.forecast_mins
    base_config.FILTERS = exp_config.filters
    base_config.KERNEL_SIZE = exp_config.kernel_size
    base_config.NUM_STATIONS = exp_config.num_stations
    
    # Load data
    print("Loading data...")
    data_gen = SubwayDataGenerator(base_config)
    data_gen.load_data(normalize=True, max_headway=30.0)
    
    # Create train/val split (80/20 temporal split)
    total_samples = len(data_gen.headway_data) - exp_config.lookback_mins - exp_config.forecast_mins
    train_end = int(total_samples * 0.8)
    
    print(f"Train samples: 0 to {train_end}")
    print(f"Val samples: {train_end} to {total_samples}")
    
    train_ds = data_gen.make_dataset(start_index=0, end_index=train_end, shuffle=True)
    val_ds = data_gen.make_dataset(start_index=train_end, end_index=None, shuffle=False)
    
    # ========================================================================
    # Start Vertex AI Experiment Run
    # ========================================================================
    print(f"\nStarting Vertex AI Experiment Run: {run_name}")
    
    # Use aiplatform.start_run() - the correct API for experiment runs
    aiplatform.start_run(run_name)
    
    try:
        # Log hyperparameters
        aiplatform.log_params({
            "exp_id": exp_id,
            "exp_name": exp_config.exp_name,
            "spatial_dropout_rate": exp_config.spatial_dropout_rate,
            "weight_decay": exp_config.weight_decay,
            "learning_rate": exp_config.learning_rate,
            "batch_size": exp_config.batch_size,
            "epochs": exp_config.epochs,
            "early_stopping_patience": exp_config.early_stopping_patience,
            "lookback_mins": exp_config.lookback_mins,
            "forecast_mins": exp_config.forecast_mins,
            "filters": exp_config.filters,
            "num_stations": exp_config.num_stations,
            "train_samples": train_end,
            "val_samples": total_samples - train_end,
        })
        print("Logged hyperparameters to Vertex AI Experiments")
        
        # Build model with regularization
        print("\nBuilding model...")
        model_builder = HeadwayConvLSTM(
            config=base_config,
            spatial_dropout_rate=exp_config.spatial_dropout_rate
        )
        model = model_builder.build_model()
        
        # Compile with AdamW for weight decay support
        print(f"Compiling with AdamW (lr={exp_config.learning_rate}, weight_decay={exp_config.weight_decay})")
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=exp_config.learning_rate,
            weight_decay=exp_config.weight_decay
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[rmse_seconds, r_squared]
        )
        
        model.summary()
        
        # Create callbacks - use local output dir + Vertex AI Experiments callback
        callbacks = create_callbacks(exp_config, local_output_dir)
        callbacks.append(VertexExperimentCallback())
        
        # Train
        print(f"\nStarting training for {exp_config.epochs} epochs...")
        print(f"  Early stopping patience: {exp_config.early_stopping_patience}")
        print(f"  Spatial dropout rate: {exp_config.spatial_dropout_rate}")
        print(f"  Weight decay: {exp_config.weight_decay}")
        print()
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=exp_config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate final metrics
        best_val_loss = float(min(history.history['val_loss']))
        best_val_rmse = float(min(history.history['val_rmse_seconds']))
        best_val_r2 = float(max(history.history['val_r_squared']))
        best_epoch = int(np.argmin(history.history['val_loss']) + 1)
        total_epochs = len(history.history['loss'])
        
        # Log final summary metrics to Vertex AI Experiments
        aiplatform.log_metrics({
            "best_val_loss": best_val_loss,
            "best_val_rmse_seconds": best_val_rmse,
            "best_val_r_squared": best_val_r2,
            "best_epoch": best_epoch,
            "total_epochs": total_epochs,
        })
        print("Logged final metrics to Vertex AI Experiments")
        
        # Save results
        results = {
            "exp_id": exp_id,
            "exp_name": exp_config.exp_name,
            "description": exp_config.description,
            "run_name": run_name,
            "config": {
                "spatial_dropout_rate": exp_config.spatial_dropout_rate,
                "weight_decay": exp_config.weight_decay,
                "learning_rate": exp_config.learning_rate,
                "batch_size": exp_config.batch_size,
            },
            "results": {
                "best_val_loss": best_val_loss,
                "best_val_rmse_seconds": best_val_rmse,
                "best_val_r_squared": best_val_r2,
                "best_epoch": best_epoch,
                "total_epochs": total_epochs,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save results JSON locally
        results_path = os.path.join(local_output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Upload to GCS if needed
        if gcs_output_dir:
            upload_to_gcs(local_output_dir, gcs_output_dir)
        
        # End of experiment run
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Experiment Run: {run_name}")
        print(f"Best Val Loss: {results['results']['best_val_loss']:.6f}")
        print(f"Best Val RMSE (seconds): {results['results']['best_val_rmse_seconds']:.2f}")
        print(f"Best Val R²: {results['results']['best_val_r_squared']:.4f}")
        print(f"Best Epoch: {results['results']['best_epoch']}")
        print(f"Results saved to: {results_path}")
        if gcs_output_dir:
            print(f"Uploaded to: {gcs_output_dir}")
        print(f"\nView in Vertex AI Console:")
        print(f"  https://console.cloud.google.com/vertex-ai/experiments/{experiment_name}?project={project}")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\nExperiment FAILED: {e}")
        aiplatform.log_params({"status": "failed", "error": str(e)})
        raise
    finally:
        # Always end the experiment run
        aiplatform.end_run()


def main():
    parser = argparse.ArgumentParser(description="Run a training experiment with Vertex AI Experiments tracking")
    parser.add_argument("--exp_id", type=int, required=True, help="Experiment ID (1-4)")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory (local or GCS path)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (local or GCS path)")
    parser.add_argument("--project", type=str, default="time-series-478616", help="GCP project ID")
    parser.add_argument("--location", type=str, default="us-east1", help="GCP region")
    parser.add_argument("--experiment", type=str, default="headway-regularization", help="Vertex AI Experiment name")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    run_experiment(
        exp_id=args.exp_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        project=args.project,
        location=args.location,
        experiment_name=args.experiment,
    )


if __name__ == "__main__":
    main()
