
import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import aiplatform
from tensorflow import keras
from tensorflow.keras import layers
from src.config import config
from src.data_utils import create_windowed_dataset, MAESeconds
from typing import Tuple, Optional

# --- Model Builder ---
class VertexAILoggingCallback(keras.callbacks.Callback):
    """Logs metrics to Vertex AI Experiments at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        try:
            if logs:
                # Cast all metrics to Python float to avoid Vertex AI TypeError with numpy types
                metrics = {k: float(v) for k, v in logs.items()}
                aiplatform.log_metrics(metrics)
        except Exception as e:
            # Silence logging errors (e.g. no active run in smoke test)
            pass
            try:
                # log_time_series_metrics handles plotting over time (step=epoch)
                aiplatform.log_time_series_metrics(metrics, step=epoch + 1)
            except Exception as e:
                print(f"Warning: Failed to log metrics to Vertex AI: {e}")

def build_model(lookback_steps: int, n_features: int, num_routes: int) -> keras.Model:
    """Builds the Stacked GRU Model with Dual Outputs."""
    
    inputs = layers.Input(shape=(lookback_steps, n_features), name='input_sequence')
    
    x = inputs
    
    # Tunable GRU Layers
    # config.gru_units is likely [64, 32]
    # We iterate, returning sequences for all but the last
    
    gru_units = config.gru_units
    dropout_rate = config.dropout_rate
    
    for i, units in enumerate(gru_units):
        return_sequences = (i < len(gru_units) - 1)
        x = layers.GRU(
            units,
            return_sequences=return_sequences,
            name=f'gru_{i}'
        )(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Dual Output Heads
    
    # 1. Regression Head (Headway)
    headway_out = layers.Dense(1, name='headway')(x)
    
    # 2. Classification Head (Route)
    route_out = layers.Dense(num_routes, activation='softmax', name='route')(x)
    
    model = keras.Model(inputs=inputs, outputs=[headway_out, route_out])
    
    return model

# --- Data Loading & Windowing ---
def create_datasets(df: pd.DataFrame, test_output_path: Optional[str] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Generates Windowed TF Datasets using shared utility logic."""
    
    n = len(df)
    train_end = int(n * config.train_split)
    val_end = int(n * (config.train_split + config.val_split))
    
    # Export Test Artifact Logic
    if test_output_path:
        print(f"Exporting test artifact to {test_output_path}...")
        # We need raw rows for the Test Set (indices val_end to end)
        df_test = df.iloc[val_end:].copy()
        print(f"Test Set Shape: {df_test.shape}")
        
        # Ensure dir exists
        os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
        df_test.to_csv(test_output_path, index=False)
        print("Test artifact exported.")

    # Use Validated Shared Logic from src.data_utils
    # This prevents training/evaluation skew
    
    batch_size = config.batch_size
    lookback_steps = config.lookback_steps
    
    print("Generating Training Dataset...")
    train_ds = create_windowed_dataset(
        df, 
        batch_size=batch_size, 
        lookback_steps=lookback_steps,
        start_index=0, 
        end_index=train_end, 
        shuffle=True
    )
    
    print("Generating Validation Dataset...")
    val_ds = create_windowed_dataset(
        df, 
        batch_size=batch_size, 
        lookback_steps=lookback_steps,
        start_index=train_end, 
        end_index=val_end, 
        shuffle=False
    )
    
    # test_ds is not used for training, only the exported CSV is used for evaluation.
    # We return None for test_ds to save resources.
    
    return train_ds, val_ds, None

# --- Training Loop ---
def train_model(input_path: str, model_output_path: str, test_data_output_path: str = None):
    print(f"Loading preprocessed data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Data Loaded. Shape: {df.shape}")
    
    # Detect num features
    # NOTE: 'scheduled_headway' is metadata, not a feature. Exclude from count if present.
    feature_cols = df.columns.tolist()
    if 'scheduled_headway' in feature_cols:
        feature_cols.remove('scheduled_headway')
    n_features = len(feature_cols)

    num_routes = len([c for c in df.columns if c.startswith('route_')])
    
    print("Creating datasets...")
    # Pass test_data_output_path to handle export during split definition
    train_ds, val_ds, _ = create_datasets(df, test_output_path=test_data_output_path)
    
    print("Building model...")
    model = build_model(
        lookback_steps=config.lookback_steps,
        n_features=n_features,
        num_routes=num_routes
    )
    
    print("Compiling model...")
    optimizer = keras.optimizers.get({
        'class_name': 'Adam', # Simple default or use config.optimizer
        'config': {'learning_rate': config.learning_rate}
    })
    
    # Losses
    # Headway: Huber, Route: CategoricalCrossentropy
    losses = {
        'headway': keras.losses.Huber(),
        'route': keras.losses.CategoricalCrossentropy()
    }
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics={
            'headway': [MAESeconds(name='mae_seconds'), 'mae'],
            'route': ['accuracy']
        }
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_output_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        ),
        VertexAILoggingCallback()
    ]
    
    # TensorBoard & Experiment Tracking
    # -------------------------------------------------------------------------
    # 1. Initialize Vertex AI Experiment
    # -------------------------------------------------------------------------
    print(f"Initializing Vertex AI Experiment: {config.experiment_name}, Run: {config.run_name}")
    try:
        aiplatform.init(
            project=config.project_id,
            location=config.region,
            experiment=config.experiment_name,
            experiment_tensorboard=config.tensorboard_resource
        )
        
        # Start the run
        aiplatform.start_run(run=config.run_name, resume=True)
    except Exception as e:
        print(f"Warning: Failed to initialize/start Vertex AI Experiment ({e}). Continuing without tracking.")
    
    # Log Hyperparameters
    params_to_log = {
        "gru_units": str(config.gru_units),
        "dropout_rate": config.dropout_rate,
        "lookback_steps": config.lookback_steps,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer,
        "train_split": config.train_split
    }
    try:
        aiplatform.log_params(params_to_log)
    except Exception:
        pass
    
    # -------------------------------------------------------------------------
    # 2. Configure TensorBoard Callback
    # -------------------------------------------------------------------------
    if config.bucket_name:
        # Construct GCS log path: gs://bucket_name/tensorboard_logs/run_name
        # Note: bucket_name in config might strictly be the name "my-bucket"
        tb_log_dir = f"gs://{config.bucket_name}/tensorboard_logs/{config.run_name}"
        print(f"TensorBoard logging enabled: {tb_log_dir}")
        
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=tb_log_dir,
                histogram_freq=config.histogram_freq if config.track_histograms else 0,
                write_graph=True,
                write_images=False
            )
        )
    else:
        print("WARNING: No GCS Bucket configured. TensorBoard logs will only be local.")

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks
    )
    
    # -------------------------------------------------------------------------
    # 3. Log Final Metrics to Experiment
    # -------------------------------------------------------------------------
    # Get the best val loss from history (or last)
    final_metrics = {}
    for metric, values in history.history.items():
        # Log the final value (or best if we added logic for that, sticking to last/min for val_loss)
        if "val_" in metric and "loss" in metric:
            final_metrics[metric] = float(min(values)) # Best val loss
        else:
            final_metrics[metric] = float(values[-1]) # Last value for others
            
    # Remove callbacks object from serialization if present (rare in metrics but good safety)
    
    print(f"Logging metrics: {final_metrics}")
    try:
        aiplatform.log_metrics(final_metrics)
        aiplatform.end_run()
    except Exception as e:
        print(f"Warning: Failed to log metrics to Vertex AI ({e}).")
    
    print(f"Saving model to {model_output_path}...")
    # KFP output path could be a file path (model.h5) or directory? 
    # Usually Artifact maps to a full path if extension provided.
    
    # Ensure dir exists
    dirname = os.path.dirname(model_output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
        
    model.save(model_output_path)
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_output_path", type=str, required=True)
    parser.add_argument("--test_data_output_path", type=str, required=False, default=None)
    
    args = parser.parse_args()
    
    # 1. Train and Export
    train_model(args.input_path, args.model_output_path, args.test_data_output_path)

