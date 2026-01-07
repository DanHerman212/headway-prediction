#!/usr/bin/env python3
"""
Baseline Training Script with Full TensorBoard Tracking

Runs the paper-faithful ConvLSTM baseline with comprehensive experiment
tracking via the new src/tracking module.

Features:
- TensorBoard: Scalars, Histograms, Graphs, HParams, Images
- Vertex AI Experiments integration
- GCS checkpoint upload
- Spatiotemporal visualization

Usage:
    # Local
    python -m src.experiments.run_baseline --local
    
    # On Vertex AI (called from pipeline)
    python -m src.experiments.run_baseline --data_dir gs://bucket/data
"""

import argparse
import os
import json
from datetime import datetime
from typing import Optional

import tensorflow as tf
import numpy as np

# Local imports
from src.models.baseline_convlstm import BaselineConvLSTM, count_params
from src.data.dataset import SubwayDataGenerator
from src.config import Config
from src.metrics import rmse_seconds, r_squared
from src.tracking import Tracker, TrackerConfig
from src.visualizations import SpatiotemporalCallback


# ============================================================================
# Constants
# ============================================================================

PROJECT = "time-series-478616"
REGION = "us-east1"
BUCKET = "st-convnet-training-configuration"
EXPERIMENT_NAME = "headway-baseline"


# ============================================================================
# Data Loading
# ============================================================================

def download_gcs_data(gcs_path: str, local_dir: str = "/tmp/data") -> str:
    """Download data from GCS to local directory."""
    from google.cloud import storage
    
    os.makedirs(local_dir, exist_ok=True)
    
    path = gcs_path.replace("gs://", "")
    bucket_name = path.split("/")[0]
    prefix = "/".join(path.split("/")[1:])
    
    print(f"Downloading from {gcs_path}...")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for filename in ["headway_matrix_full.npy", "schedule_matrix_full.npy"]:
        blob_path = f"{prefix}/{filename}" if prefix else filename
        blob = bucket.blob(blob_path)
        local_path = os.path.join(local_dir, filename)
        
        blob.download_to_filename(local_path)
        print(f"  Downloaded: {filename}")
    
    return local_dir


def upload_to_gcs(local_dir: str, gcs_dir: str):
    """Upload directory contents to GCS."""
    from google.cloud import storage
    
    path = gcs_dir.replace("gs://", "")
    bucket_name = path.split("/")[0]
    prefix = "/".join(path.split("/")[1:])
    
    print(f"Uploading to {gcs_dir}...")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_path = f"{prefix}/{rel_path}" if prefix else rel_path
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
    
    print("Upload complete!")


# ============================================================================
# Training
# ============================================================================

def run_baseline(
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 32,
    local_mode: bool = False,
    enable_profiling: bool = False,
    enable_visualizations: bool = True,
):
    """
    Run paper-faithful baseline training with full tracking.
    
    Args:
        data_dir: Data directory (local or gs://)
        output_dir: Output directory for checkpoints (local or gs://)
        epochs: Training epochs (default: 100 per paper)
        batch_size: Batch size (default: 32 per paper)
        local_mode: If True, skip Vertex AI integration
        enable_profiling: Enable GPU profiling (adds overhead)
        enable_visualizations: Enable spatiotemporal image logging
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"baseline-{timestamp}"
    
    print("=" * 70)
    print("BASELINE CONVLSTM TRAINING")
    print("Paper: Usama & Koutsopoulos (2025) - arXiv:2510.03121")
    print("=" * 70)
    print(f"Run: {run_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print()
    
    # ========================================================================
    # Setup paths
    # ========================================================================
    if data_dir is None:
        data_dir = "data"
    
    if output_dir is None:
        if local_mode:
            output_dir = f"outputs/baseline/{timestamp}"
        else:
            output_dir = f"gs://{BUCKET}/runs/{run_name}"
    
    # Handle GCS paths
    actual_data_dir = data_dir
    if data_dir.startswith("gs://"):
        actual_data_dir = download_gcs_data(data_dir)
    
    gcs_output_dir = None
    if output_dir.startswith("gs://"):
        gcs_output_dir = output_dir
        local_output_dir = f"/tmp/{run_name}"
    else:
        local_output_dir = output_dir
    
    os.makedirs(local_output_dir, exist_ok=True)
    tensorboard_dir = os.path.join(local_output_dir, "tensorboard")
    
    print(f"Data: {actual_data_dir}")
    print(f"Output: {local_output_dir}")
    if gcs_output_dir:
        print(f"Upload to: {gcs_output_dir}")
    print()
    
    # ========================================================================
    # Initialize Vertex AI (if not local)
    # ========================================================================
    if not local_mode:
        try:
            from google.cloud import aiplatform
            aiplatform.init(
                project=PROJECT,
                location=REGION,
                experiment=EXPERIMENT_NAME
            )
            aiplatform.start_run(run_name)
            print(f"Vertex AI Experiment: {EXPERIMENT_NAME}/{run_name}")
        except Exception as e:
            print(f"Warning: Vertex AI init failed: {e}")
            local_mode = True
    
    # ========================================================================
    # Load data
    # ========================================================================
    print("Loading data...")
    
    config = Config()
    config.DATA_DIR = actual_data_dir
    config.BATCH_SIZE = batch_size
    config.LOOKBACK_MINS = 30  # Paper value
    config.FORECAST_MINS = 15  # Paper value
    config.NUM_STATIONS = 66
    
    data_gen = SubwayDataGenerator(config)
    data_gen.load_data(normalize=True, max_headway=30.0)
    
    # Temporal train/val split (80/20)
    total_samples = len(data_gen.headway_data) - 30 - 15
    train_end = int(total_samples * 0.8)
    
    print(f"Train samples: {train_end}")
    print(f"Val samples: {total_samples - train_end}")
    
    train_ds = data_gen.make_dataset(
        start_index=0, 
        end_index=train_end, 
        shuffle=True
    )
    val_ds = data_gen.make_dataset(
        start_index=train_end, 
        end_index=None, 
        shuffle=False
    )
    
    # ========================================================================
    # Build model
    # ========================================================================
    print("\nBuilding paper-faithful baseline model...")
    
    builder = BaselineConvLSTM.from_paper_config(n_stations=config.NUM_STATIONS)
    model = builder.build_model()
    
    # Compile with paper settings
    compile_args = builder.get_compile_args()
    compile_args['metrics'] = [rmse_seconds, r_squared]
    model.compile(**compile_args)
    
    params = count_params(model)
    print(f"Parameters: {params['trainable']:,} trainable")
    model.summary()
    
    # ========================================================================
    # Setup tracking
    # ========================================================================
    print("\nInitializing tracking...")
    
    # Hyperparameters for tracking
    hparams = {
        'model': 'BaselineConvLSTM',
        'filters': builder.filters,
        'kernel_size': str(builder.kernel_size),
        'lookback': builder.lookback,
        'forecast': builder.forecast,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss': 'mse',
        'n_stations': config.NUM_STATIONS,
        'train_samples': train_end,
        'val_samples': total_samples - train_end,
        'trainable_params': params['trainable'],
    }
    
    tracker_config = TrackerConfig(
        experiment_name=EXPERIMENT_NAME,
        run_name=run_name,
        log_dir=tensorboard_dir,
        scalars=True,
        histograms=True,
        histogram_freq=5,  # Every 5 epochs to reduce overhead
        graphs=True,
        hparams=True,
        profiling=enable_profiling,
        profile_batch_range=(10, 15),
        hparams_dict=hparams,
        description="Paper-faithful baseline ConvLSTM (arXiv:2510.03121)"
    )
    
    tracker = Tracker(tracker_config)
    
    # Log hyperparameters to Vertex AI
    if not local_mode:
        try:
            from google.cloud import aiplatform
            aiplatform.log_params(hparams)
        except Exception as e:
            print(f"Warning: Failed to log params: {e}")
    
    # ========================================================================
    # Create callbacks
    # ========================================================================
    callbacks = tracker.keras_callbacks()
    
    # Early stopping (paper: patience=50)
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    ))
    
    # LR reduction
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ))
    
    # Model checkpointing
    checkpoint_path = os.path.join(local_output_dir, "best_model.h5")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ))
    
    # Spatiotemporal visualizations
    if enable_visualizations:
        viz_callback = SpatiotemporalCallback(
            tracker=tracker,
            validation_data=val_ds,
            freq=10,  # Every 10 epochs
            num_samples=2,
            max_headway=30.0
        )
        callbacks.append(viz_callback)
    
    # Vertex AI metrics callback
    if not local_mode:
        class VertexCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    try:
                        from google.cloud import aiplatform
                        aiplatform.log_time_series_metrics({
                            k: float(v) for k, v in logs.items()
                        }, step=epoch + 1)
                    except:
                        pass
        
        callbacks.append(VertexCallback())
    
    # ========================================================================
    # Train
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # ========================================================================
    # Results
    # ========================================================================
    best_val_loss = float(min(history.history['val_loss']))
    best_val_rmse = float(min(history.history['val_rmse_seconds']))
    best_val_r2 = float(max(history.history['val_r_squared']))
    best_epoch = int(np.argmin(history.history['val_loss']) + 1)
    total_epochs_run = len(history.history['loss'])
    
    results = {
        'run_name': run_name,
        'model': 'BaselineConvLSTM',
        'paper': 'arXiv:2510.03121',
        'hyperparameters': hparams,
        'results': {
            'best_val_loss': best_val_loss,
            'best_val_rmse_seconds': best_val_rmse,
            'best_val_r_squared': best_val_r2,
            'best_epoch': best_epoch,
            'total_epochs': total_epochs_run,
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save results
    results_path = os.path.join(local_output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log final metrics to Vertex AI
    if not local_mode:
        try:
            from google.cloud import aiplatform
            aiplatform.log_metrics(results['results'])
        except:
            pass
    
    # Log hparams metrics for TensorBoard
    tracker.log_hparams_metrics({
        'hparam/best_val_loss': best_val_loss,
        'hparam/best_val_rmse_seconds': best_val_rmse,
        'hparam/best_val_r_squared': best_val_r2,
    })
    
    # Close tracker
    tracker.close()
    
    # Upload to GCS
    if gcs_output_dir:
        upload_to_gcs(local_output_dir, gcs_output_dir)
    
    # End Vertex AI run
    if not local_mode:
        try:
            from google.cloud import aiplatform
            aiplatform.end_run()
        except:
            pass
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Run: {run_name}")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Best Val RMSE: {best_val_rmse:.2f} seconds")
    print(f"Best Val R²: {best_val_r2:.4f}")
    print(f"Best Epoch: {best_epoch}/{total_epochs_run}")
    print()
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results: {results_path}")
    print(f"TensorBoard: tensorboard --logdir={tensorboard_dir}")
    if gcs_output_dir:
        print(f"GCS: {gcs_output_dir}")
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train paper-faithful baseline ConvLSTM"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=None,
        help="Data directory (local or gs://)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory (local or gs://)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Training epochs (default: 100 per paper)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size (default: 32 per paper)"
    )
    parser.add_argument(
        "--local", 
        action="store_true",
        help="Local mode (skip Vertex AI)"
    )
    parser.add_argument(
        "--profile", 
        action="store_true",
        help="Enable GPU profiling"
    )
    parser.add_argument(
        "--no_viz", 
        action="store_true",
        help="Disable spatiotemporal visualizations"
    )
    
    args = parser.parse_args()
    
    run_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        local_mode=args.local,
        enable_profiling=args.profile,
        enable_visualizations=not args.no_viz,
    )


if __name__ == "__main__":
    main()
