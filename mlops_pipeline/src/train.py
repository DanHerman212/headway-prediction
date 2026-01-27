
import argparse
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import aiplatform
from tensorflow import keras
from src.data_utils import create_windowed_dataset
from src.metrics import MAESeconds
from src.model import build_model
from src.callbacks import VertexAILoggingCallback
from src.constants import ROUTE_COL_PREFIX, SCHEDULED_HEADWAY_COL
from typing import Tuple, Optional

# --- Data Loading & Windowing ---
def create_datasets(df: pd.DataFrame, test_output_path: Optional[str] = None, cfg: DictConfig = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Generates Windowed TF Datasets using shared utility logic."""
    
    n = len(df)
    train_n = int(n * cfg.data.train_split)
    val_n = int(n * cfg.data.val_split)
    
    train_end = train_n
    val_end = train_n + val_n
    
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

    batch_size = cfg.training.batch_size
    lookback_steps = cfg.model.lookback_steps
    
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
    
    return train_ds, val_ds, None

# --- Training Loop ---
def train_model(cfg: DictConfig):
    input_path = cfg.paths.input_path
    model_output_path = cfg.paths.model_output_path
    test_data_output_path = cfg.paths.test_data_output_path
    
    print(f"Loading preprocessed data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Data Loaded. Shape: {df.shape}")
    
    # Detect num features
    feature_cols = df.columns.tolist()
    if SCHEDULED_HEADWAY_COL in feature_cols:
        feature_cols.remove(SCHEDULED_HEADWAY_COL)
    
    # SAFETY: Only count numeric features to match data_utils.py logic
    # We must filter the dataframe first to know the true column count
    numeric_df = df.select_dtypes(include=[np.number])
    # Also drop scheduled_headway from this numeric view if it exists
    if SCHEDULED_HEADWAY_COL in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[SCHEDULED_HEADWAY_COL])
        
    n_features = len(numeric_df.columns)

    num_routes = len([c for c in df.columns if c.startswith(ROUTE_COL_PREFIX)])
    
    print("Creating datasets...")
    train_ds, val_ds, _ = create_datasets(df, test_output_path=test_data_output_path, cfg=cfg)
    
    print("Building model...")
    model = build_model(
        lookback_steps=cfg.model.lookback_steps,
        n_features=n_features,
        num_routes=num_routes,
        cfg=cfg
    )
    
    print("Compiling model...")
    optimizer = keras.optimizers.get({
        'class_name': 'Adam', 
        'config': {'learning_rate': cfg.training.learning_rate}
    })
    
    # Losses
    losses = {
        'headway': cfg.training.loss_function,
        'route': 'categorical_crossentropy'
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
        )
    ]

    # Offline Mode Check
    is_offline = cfg.experiment.get("offline", False)
    
    if not is_offline:
        callbacks.append(VertexAILoggingCallback())

    # TensorBoard & Experiment Tracking
    # run_name is usually passed via env var (overridden by pipeline) or config
    run_name = os.environ.get("RUN_NAME", cfg.experiment.run_name)

    if not is_offline:
        print(f"Initializing Vertex AI Experiment: {cfg.experiment.name}, Run: {run_name}")
        try:
            aiplatform.init(
                project=cfg.experiment.project_id,
                location=cfg.experiment.location,
                experiment=cfg.experiment.name,
                experiment_tensorboard=cfg.experiment.tensorboard_resource_name
            )
            
            print(f"Attempting to create run: {run_name}")
            try:
                 aiplatform.start_run(run=run_name, resume=False)
                 print(f"Successfully created run: {run_name}")
            except Exception as e:
                 if "409" in str(e) or "AlreadyExists" in str(e):
                     print(f"Run {run_name} already exists. Resuming...")
                     aiplatform.start_run(run=run_name, resume=True)
                     print(f"Successfully resumed run: {run_name}")
                 else:
                     print(f"Failed to create run ({e}). Attempting resume...")
                     aiplatform.start_run(run=run_name, resume=True)

        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize/start Vertex AI Experiment: {e}")
            raise e
        
        # Log Hyperparameters
        params_to_log = OmegaConf.to_container(cfg.model, resolve=True)
        params_to_log.update(OmegaConf.to_container(cfg.training, resolve=True))
        params_to_log.update(OmegaConf.to_container(cfg.data, resolve=True))

        try:
            flat_params = {}
            for k, v in params_to_log.items():
                flat_params[k] = str(v) if isinstance(v, (list, dict)) else v
                
            aiplatform.log_params(flat_params)
        except Exception:
            pass
    else:
        print(f"Offline mode: Skipping Vertex AI Experiment tracking for run {run_name}")
    
    # 2. Configure TensorBoard Callback
    if cfg.experiment.artifact_bucket and not is_offline:
        tb_log_dir = f"gs://{cfg.experiment.artifact_bucket}/tensorboard_logs/{run_name}"
        print(f"TensorBoard logging enabled: {tb_log_dir}")
        
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=tb_log_dir,
                histogram_freq=cfg.experiment.histogram_freq if cfg.experiment.track_histograms else 0,
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
        epochs=cfg.training.epochs,
        callbacks=callbacks
    )
    
    # 3. Log Final Metrics
    final_metrics = {}
    for metric, values in history.history.items():
        if "val_" in metric and "loss" in metric:
            final_metrics[metric] = float(min(values)) 
        else:
            final_metrics[metric] = float(values[-1]) 
            
    print(f"Logging metrics: {final_metrics}")
    
    if not is_offline:
        try:
            aiplatform.log_metrics(final_metrics)
            aiplatform.end_run()
        except Exception as e:
            print(f"Warning: Failed to log metrics to Vertex AI ({e}).")
    
    print(f"Saving model to {model_output_path}...")
    dirname = os.path.dirname(model_output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
        
    model.save(model_output_path)
    print("Training Complete.")

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train_model(cfg)

if __name__ == "__main__":
    main()

