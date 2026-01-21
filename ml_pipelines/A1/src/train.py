"""
Training Module for A1 Track Model

Handles:
- Loading preprocessed data
- Chronological train/val/test splitting
- Creating tf.data.Dataset using timeseries_dataset_from_array
- Training with TensorBoard and Vertex AI Experiments integration
- Saving model artifacts

Usage:
    python train.py --run_name exp01-baseline
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import config
from .model import get_model, mae_seconds


def load_preprocessed_data(data_path: str = None) -> Tuple[np.ndarray, Dict]:
    """
    Load preprocessed numpy array and metadata.
    
    Args:
        data_path: Path to preprocessed .npy file
    
    Returns:
        Tuple of (data_array, metadata_dict)
    """
    if data_path is None:
        data_path = config.preprocessed_data_path
    
    print(f"Loading preprocessed data from: {data_path}")
    # Standard numpy array - no pickle needed
    X = np.load(data_path)
    print(f"  Shape: {X.shape}")
    
    # Load metadata (next to the .npy file)
    metadata_path = data_path.replace('.npy', '_metadata.json')
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return X, metadata


def calculate_split_indices(n_samples: int) -> Tuple[int, int, int]:
    """
    Calculate chronological split indices for train/val/test.
    
    Args:
        n_samples: Total number of samples
    
    Returns:
        Tuple of (train_end, val_end, test_end)
    """
    train_end = int(n_samples * config.TRAIN_SPLIT)
    val_end = int(n_samples * (config.TRAIN_SPLIT + config.VAL_SPLIT))
    test_end = n_samples
    
    print(f"\nChronological data splits:")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Train: 0 to {train_end:,} ({config.TRAIN_SPLIT*100:.0f}%)")
    print(f"  Val: {train_end:,} to {val_end:,} ({config.VAL_SPLIT*100:.0f}%)")
    print(f"  Test: {val_end:,} to {test_end:,} ({config.TEST_SPLIT*100:.0f}%)")
    
    return train_end, val_end, test_end


def create_timeseries_datasets(
    X: np.ndarray,
    train_end: int,
    val_end: int,
    test_end: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create tf.data.Dataset for train/val/test using timeseries_dataset_from_array.
    
    Each sample contains:
    - Input: [lookback_window] timesteps of features
    - Targets: next timestep's [route_one_hot, log_headway]
    
    Args:
        X: Preprocessed feature array (n_samples, n_features)
        train_end: End index for training data
        val_end: End index for validation data
        test_end: End index for test data
    
    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    print(f"\nCreating timeseries datasets...")
    print(f"  Lookback window: {config.LOOKBACK_WINDOW}")
    print(f"  Forecast horizon: {config.FORECAST_HORIZON}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    
    # Extract features for input and targets
    # Features: [log_headway, route_A, route_C, route_E, hour_sin, hour_cos, dow_sin, dow_cos]
    # Target route: indices 1-3 (route_A, route_C, route_E)
    # Target headway: index 0 (log_headway)
    
    def create_dataset(start_idx: int, end_idx: int, shuffle: bool = False) -> tf.data.Dataset:
        """Helper to create dataset for a specific split."""
        
        # Input sequences
        X_split = X[start_idx:end_idx]
        
        # Create targets (next timestep after each sequence)
        # Route: one-hot [route_A, route_C, route_E]
        route_targets = X[start_idx + config.LOOKBACK_WINDOW:end_idx + config.LOOKBACK_WINDOW, 1:4]
        # Headway: log_headway
        headway_targets = X[start_idx + config.LOOKBACK_WINDOW:end_idx + config.LOOKBACK_WINDOW, 0:1]
        
        # Adjust array length to match (we need lookback_window + 1 to create target)
        max_samples = len(X_split) - config.LOOKBACK_WINDOW
        X_split = X_split[:max_samples + config.LOOKBACK_WINDOW]
        
        print(f"    Creating dataset: samples={max_samples:,}, shuffle={shuffle}")
        
        # Create timeseries dataset
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=X_split,
            targets=None,  # We'll manually add targets
            sequence_length=config.LOOKBACK_WINDOW,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=config.BATCH_SIZE,
        )
        
        # Add targets as separate dataset
        target_dataset = tf.data.Dataset.from_tensor_slices({
            'route_output': route_targets,
            'headway_output': headway_targets
        }).batch(config.BATCH_SIZE)
        
        # Zip inputs and targets
        dataset = tf.data.Dataset.zip((dataset, target_dataset))
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    # Create splits
    train_ds = create_dataset(0, train_end, shuffle=True)
    val_ds = create_dataset(train_end, val_end, shuffle=False)
    test_ds = create_dataset(val_end, test_end, shuffle=False)
    
    print(f"  ✓ Datasets created")
    
    return train_ds, val_ds, test_ds


def create_callbacks(run_name: str) -> list:
    """
    Create training callbacks including TensorBoard with profiling.
    
    Args:
        run_name: Unique identifier for this training run
    
    Returns:
        List of Keras callbacks
    """
    print(f"\nConfiguring callbacks for run: {run_name}")
    
    callbacks = []
    
    # 1. TensorBoard with full tracking
    tensorboard_dir = os.path.join(config.TENSORBOARD_LOG_DIR, run_name)
    print(f"  TensorBoard log dir: {tensorboard_dir}")
    
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=config.HISTOGRAM_FREQ if config.TRACK_HISTOGRAMS else 0,
        write_graph=config.TRACK_GRAPH,
        write_images=False,
        update_freq='epoch',
        profile_batch=(config.PROFILE_BATCH_RANGE if config.TRACK_PROFILING else 0),
        embeddings_freq=0
    )
    callbacks.append(tensorboard_callback)
    print(f"    ✓ TensorBoard (scalars, histograms, graph, profiling)")
    
    # 2. Model checkpoint
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = config.checkpoint_path
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=config.MONITOR_METRIC,
        mode=config.CHECKPOINT_MODE,
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    print(f"    ✓ ModelCheckpoint: {checkpoint_path}")
    
    # 3. Early stopping
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor=config.MONITOR_METRIC,
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping_callback)
    print(f"    ✓ EarlyStopping (patience={config.EARLY_STOPPING_PATIENCE})")
    
    # 4. Reduce learning rate on plateau
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor=config.MONITOR_METRIC,
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=1
    )
    callbacks.append(reduce_lr_callback)
    print(f"    ✓ ReduceLROnPlateau (factor={config.REDUCE_LR_FACTOR}, patience={config.REDUCE_LR_PATIENCE})")
    
    # 5. CSV Logger
    csv_log_path = os.path.join(config.CHECKPOINT_DIR, f"{run_name}_training_log.csv")
    csv_callback = keras.callbacks.CSVLogger(csv_log_path)
    callbacks.append(csv_callback)
    print(f"    ✓ CSVLogger: {csv_log_path}")
    
    return callbacks


def train_model(run_name: str = None, use_vertex_experiments: bool = True) -> Dict:
    """
    Complete training pipeline.
    
    Args:
        run_name: Unique identifier for this run
        use_vertex_experiments: Whether to track with Vertex AI Experiments
    
    Returns:
        Dictionary with training history and metadata
    """
    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    print("="*80)
    print(f"Training A1 Model - Run: {run_name}")
    print("="*80)
    
    # Initialize Vertex AI Experiments (if enabled)
    vertex_run = None
    if use_vertex_experiments:
        try:
            from google.cloud import aiplatform
            
            # Initialize with experiment specified
            aiplatform.init(
                project=config.BQ_PROJECT,
                location=config.BQ_LOCATION,
                experiment=config.EXPERIMENT_NAME
            )
            
            # Start run (tensorboard integration via callbacks, not here)
            vertex_run = aiplatform.start_run(run=run_name)
            
            # Log hyperparameters
            vertex_run.log_params(config.hparams_dict)
            
            print(f"\n✓ Vertex AI Experiment: {config.EXPERIMENT_NAME}")
            print(f"  Run: {run_name}")
            
        except Exception as e:
            print(f"\n⚠️  Vertex AI Experiments not available: {e}")
            print("  Continuing with local training only...")
    
    # Load data
    X, metadata = load_preprocessed_data()
    
    # Calculate splits
    train_end, val_end, test_end = calculate_split_indices(X.shape[0])
    
    # Create datasets
    train_ds, val_ds, test_ds = create_timeseries_datasets(X, train_end, val_end, test_end)
    
    # Build model
    print("\n" + "="*80)
    model = get_model(compile=True)
    print("="*80)
    
    # Create callbacks
    callbacks = create_callbacks(run_name)
    
    # Train model
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Log final metrics to Vertex AI
    if vertex_run:
        final_metrics = {
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_train_route_accuracy': float(history.history['route_output_accuracy'][-1]),
            'final_val_route_accuracy': float(history.history['val_route_output_accuracy'][-1]),
            'final_train_headway_mae': float(history.history['headway_output_mae_seconds'][-1]),
            'final_val_headway_mae': float(history.history['val_headway_output_mae_seconds'][-1]),
        }
        vertex_run.log_metrics(final_metrics)
        vertex_run.end_run()
    
    # Save training history
    history_path = os.path.join(config.CHECKPOINT_DIR, f"{run_name}_history.json")
    with open(history_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        json.dump(history_dict, f, indent=2)
    print(f"\n✓ Training history saved: {history_path}")
    
    # Save config snapshot
    config_path = os.path.join(config.CHECKPOINT_DIR, f"{run_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"✓ Config snapshot saved: {config_path}")
    
    return {
        'run_name': run_name,
        'history': history.history,
        'model_path': config.checkpoint_path,
        'tensorboard_dir': os.path.join(config.TENSORBOARD_LOG_DIR, run_name)
    }


def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(description='Train A1 track model')
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Unique run identifier (default: auto-generated timestamp)'
    )
    parser.add_argument(
        '--no_vertex',
        action='store_true',
        help='Disable Vertex AI Experiments tracking'
    )
    
    args = parser.parse_args()
    
    result = train_model(
        run_name=args.run_name,
        use_vertex_experiments=not args.no_vertex
    )
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nRun: {result['run_name']}")
    print(f"Model: {result['model_path']}")
    print(f"TensorBoard: {result['tensorboard_dir']}")
    print(f"\nView in TensorBoard:")
    print(f"  tensorboard --logdir {config.TENSORBOARD_LOG_DIR}")
    print(f"\nNext step: Evaluate model")
    print(f"  python evaluate.py --run_name {result['run_name']}")


if __name__ == "__main__":
    main()
