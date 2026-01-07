#!/usr/bin/env python3
"""
Single entry point for running training experiments on Vertex AI.

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

from src.experiments.experiment_config import get_experiment, list_experiments, ExperimentConfig
from src.models.st_convnet import HeadwayConvLSTM
from src.data.dataset import SubwayDataGenerator
from src.config import Config
from src.metrics import rmse_seconds, r_squared


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


def create_callbacks(config: ExperimentConfig):
    """Create training callbacks with TensorBoard and GCS checkpointing."""
    
    os.makedirs(config.experiment_output_dir, exist_ok=True)
    
    # Paths
    checkpoint_path = os.path.join(config.experiment_output_dir, "best_model.keras")
    tensorboard_dir = os.path.join(config.experiment_output_dir, "tensorboard")
    
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
        # Model checkpointing
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
    ]
    
    return callbacks


def run_experiment(exp_id: int, data_dir: str = None, output_dir: str = None):
    """
    Run a single training experiment.
    
    Args:
        exp_id: Experiment ID (1-4)
        data_dir: Override data directory (for GCS paths)
        output_dir: Override output directory (for GCS paths)
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
    
    # If data is on GCS, download to local /tmp
    actual_data_dir = exp_config.data_dir
    if actual_data_dir.startswith("gs://"):
        actual_data_dir = download_gcs_data(exp_config.data_dir)
    
    print(f"Loading data from {actual_data_dir}...")
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
    
    # Create callbacks
    callbacks = create_callbacks(exp_config)
    
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
    
    # Save results
    results = {
        "exp_id": exp_id,
        "exp_name": exp_config.exp_name,
        "description": exp_config.description,
        "config": {
            "spatial_dropout_rate": exp_config.spatial_dropout_rate,
            "weight_decay": exp_config.weight_decay,
            "learning_rate": exp_config.learning_rate,
            "batch_size": exp_config.batch_size,
        },
        "results": {
            "best_val_loss": float(min(history.history['val_loss'])),
            "best_val_rmse_seconds": float(min(history.history['val_rmse_seconds'])),
            "best_val_r_squared": float(max(history.history['val_r_squared'])),
            "best_epoch": int(np.argmin(history.history['val_loss']) + 1),
            "total_epochs": len(history.history['loss']),
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results JSON
    results_path = os.path.join(exp_config.experiment_output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Best Val Loss: {results['results']['best_val_loss']:.6f}")
    print(f"Best Val RMSE (seconds): {results['results']['best_val_rmse_seconds']:.2f}")
    print(f"Best Val R²: {results['results']['best_val_r_squared']:.4f}")
    print(f"Best Epoch: {results['results']['best_epoch']}")
    print(f"Results saved to: {results_path}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--exp_id", type=int, required=True, help="Experiment ID (1-4)")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory (local or GCS path)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (local or GCS path)")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    # Setup GCS if needed
    if args.data_dir and args.data_dir.startswith("gs://"):
        setup_gcs_auth()
    
    run_experiment(
        exp_id=args.exp_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
