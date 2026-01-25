"""Model training for headway prediction."""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ml_pipelines.config.model_config import ModelConfig

from ml_pipelines.evaluation.metrics import MAESeconds


class Trainer:
    """Handles model training with time series data."""
    
    def __init__(self, config: "ModelConfig"):
        """
        Initialize trainer.
        
        Args:
            config: ModelConfig instance with training parameters
        """
        self.config = config
        self.data = None  # Store full dataframe
        self.input_x = None
        self.input_t = None
        self.input_r = None
    
    def load_data(self, data_path: str) -> None:
        """
        Load preprocessed data and create input arrays.
        
        Args:
            data_path: Path to X.csv (preprocessed features)
        """
        # Load preprocessed data
        self.data = pd.read_csv(data_path)
        
        # Create input arrays
        self.input_x = self.data.values
        self.input_t = self.data['log_headway'].values
        self.input_r = self.data[['route_A', 'route_C', 'route_E']].values
    
    def _get_split_indices(self) -> Tuple[int, int]:
        """Calculate train/val split indices based on data length."""
        n = len(self.input_x)
        train_end = int(n * self.config.train_split)
        val_end = int(n * (self.config.train_split + self.config.val_split))
        return train_end, val_end

    def save_test_set(self, output_path: str) -> None:
        """
        Save the test split of the data to CSV.
        
        Args:
            output_path: Path to save the test dataset (KFP Artifact path)
        """
        if self.data is None:
            raise ValueError("Must call load_data() before save_test_set()")
            
        _, val_end = self._get_split_indices()
        df_test = self.data.iloc[val_end:]
        
        print(f"DEBUG: Output Path provided by KFP: {output_path}")

        # FORCE NATIVE GCS UPLOAD
        # Relying on GCS Fuse (file system) is causing consistency errors between pods.
        # We must push to the cloud API directly to guarantee visibility for the next step.
        if output_path.startswith('/gcs/'):
            try:
                from google.cloud import storage
                print("DEBUG: Detected GCS path. Switching to Native Client for atomic upload.")
                
                # Clean path: /gcs/bucket/path -> bucket, path
                path_parts = output_path[5:].split('/')
                bucket_name = path_parts[0]
                # Reconstruct the blob prefix
                prefix = '/'.join(path_parts[1:])
                
                # Check if KFP gave us a file path or a dir path
                if output_path.endswith('.csv'):
                    blob_name = prefix
                else:
                    blob_name = f"{prefix}/test_data.csv"
                
                print(f"DEBUG: Uploading to gs://{bucket_name}/{blob_name}")
                
                # Write locally first
                local_tmp = '/tmp/test_data_upload.csv'
                df_test.to_csv(local_tmp, index=False)
                
                # Push
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_tmp)
                
                print("SUCCESS: Native Upload Complete.")
                return
            except Exception as e:
                print(f"CRITICAL WARNING: Native upload failed: {e}. Falling back to FS.")

        # Fallback (Local Test or Fuse Failure)
        if output_path.endswith('.csv'):
            final_path = output_path
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
        else:
            os.makedirs(output_path, exist_ok=True)
            final_path = os.path.join(output_path, 'test_data.csv')

        print(f"DEBUG: Saving test data locally to: {final_path}")
        df_test.to_csv(final_path, index=False)
        print("DEBUG: File write complete.")

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create train/val/test datasets using index slicing.
        Replaces timeseries_dataset_from_array to avoid TF Graph crashes.
        
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        if self.input_x is None:
            raise ValueError("Must call load_data() before create_datasets()")

        # Prepare data: cast to float32 and convert to TF constants
        input_x_tf = tf.constant(self.input_x.astype(np.float32))
        
        # Prepare targets
        input_t_reshaped = self.input_t.reshape(-1, 1)
        targets_combined = np.column_stack([input_t_reshaped, self.input_r]).astype(np.float32)
        targets_tf = tf.constant(targets_combined)

        sequence_length = self.config.lookback_steps
        batch_size = self.config.batch_size
        n = len(self.input_x)

        # Define slicing logic
        @tf.function
        def get_sequence(index):
            """Get input window and target for given index."""
            # Input window: from index to index + sequence_length
            x_window = input_x_tf[index : index + sequence_length]
            
            # Target: at index + sequence_length (predict next step after window)
            y_val = targets_tf[index + sequence_length]
            
            # Split targets: (headway, route)
            return x_window, (y_val[0:1], y_val[1:4])

        # Helper to build dataset from indices
        def build_from_indices(start_idx, end_idx, is_training):
            """Build dataset from index range."""
            # Calculate valid range: ensure index + sequence_length <= n
            max_possible_idx = n - sequence_length
            
            # Cap the end_idx at max possible
            actual_end = max_possible_idx if end_idx is None else min(end_idx, max_possible_idx)
            
            if start_idx >= actual_end:
                raise ValueError(f"Invalid split: start ({start_idx}) >= end ({actual_end})")

            # Create dataset of indices (lightweight)
            ds = tf.data.Dataset.range(start_idx, actual_end)
            
            if is_training:
                ds = ds.shuffle(buffer_size=10000)

            # Map indices to data slices
            ds = ds.map(get_sequence, num_parallel_calls=tf.data.AUTOTUNE)
            
            # Batch and prefetch
            ds = ds.batch(batch_size, drop_remainder=is_training)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            
            return ds

        # Calculate splits
        train_end, val_end = self._get_split_indices()
        
        # Create datasets
        train_dataset = build_from_indices(0, train_end, is_training=True)
        val_dataset = build_from_indices(train_end, val_end, is_training=False)
        test_dataset = build_from_indices(val_end, None, is_training=False)

        return train_dataset, val_dataset, test_dataset
    
    def compile_model(self, model: keras.Model) -> keras.Model:
        """
        Compile model with multi-output losses and metrics from config.
        
        Args:
            model: Uncompiled Keras model
        
        Returns:
            Compiled Keras model
        """
        # Create optimizer from config
        optimizer = keras.optimizers.get({
            'class_name': self.config.optimizer,
            'config': {'learning_rate': self.config.learning_rate}
        })
        
        # Create loss objects
        regression_loss = keras.losses.Huber(delta=self.config.huber_delta)
        # We use CategoricalCrossentropy because targets are one-hot encoded (shape N, 3)
        classification_loss = keras.losses.CategoricalCrossentropy()
        
        # Compile
        model.compile(
            optimizer=optimizer,
            loss={'headway': regression_loss, 'route': classification_loss},
            loss_weights=self.config.loss_weights,
            metrics={
                'headway': [MAESeconds(name='mae_seconds'), 'mae'],
                'route': ['accuracy']
            }
        )
        
        return model
    
    def train(self, model: keras.Model, callbacks: List[keras.callbacks.Callback]) -> keras.callbacks.History:
        """
        Train the model with config-driven parameters.
        
        Args:
            model: Uncompiled Keras model
            callbacks: Keras callbacks (typically from ExperimentTracker)
        
        Returns:
            Training history
        """
        if self.input_x is None:
            raise ValueError("Must call load_data() before train()")
        
        # Compile model
        model = self.compile_model(model)
        
        # Create datasets
        train_dataset, val_dataset, _ = self.create_datasets()
        
        # Combine with default callbacks
        all_callbacks = self._create_default_callbacks() + (callbacks or [])
        
        # Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
            callbacks=all_callbacks,
            verbose=1
        )
        
        return history
    
    def _create_default_callbacks(self) -> List[keras.callbacks.Callback]:
        """
        Create default callbacks for training.
        
        Returns:
            List of default callbacks
        """
        callbacks = []
        
        # Model checkpoint - save best model
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=f"{self.config.checkpoint_dir}/best_model.h5",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
        
        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Reduce learning rate on plateau
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_learning_rate,
                verbose=1
            )
        )
        
        return callbacks


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """
    Main training function with Vertex AI tracking.
    
    Usage:
        python -m training.train --input_csv ... --model_dir ...
    """
    import os
    import argparse
    import sys
    from datetime import datetime
    
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from ml_pipelines.config import ModelConfig, TrackingConfig
    from ml_pipelines.models.gru_model import StackedGRUModel
    from ml_pipelines.training.train import Trainer
    from ml_pipelines.tracking.tracker import ExperimentTracker

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to preprocessed input data")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Path to save the test dataset for evaluation")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch size")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="GCS path for TensorBoard logs")
    parser.add_argument("--tensorboard_resource_name", type=str, default=None, help="Vertex AI TensorBoard resource name")
    parser.add_argument("--run_name", type=str, default=None, help="Experiment run name")
    parser.add_argument("--dropout_rate", type=float, default=None, help="Dropout rate")
    parser.add_argument("--gru_units", type=str, default=None, help="GRU units (comma separated)")
    parser.add_argument("--lookback_steps", type=int, default=None, help="Lookback steps")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    args = parser.parse_args()
    
    # 1. Load configuration
    print("="*70)
    print("LOADING CONFIGURATION")
    print("="*70)
    config = ModelConfig.from_env()
    
    # Override from args
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.dropout_rate is not None:
        config.dropout_rate = args.dropout_rate
    if args.gru_units:
        # Parse string "128,64" -> [128, 64]
        try:
            config.gru_units = [int(u) for u in args.gru_units.split(',')]
        except ValueError:
            print(f"Warning: Could not parse gru_units '{args.gru_units}'. Using default.")
    if args.lookback_steps:
        config.lookback_steps = args.lookback_steps
    if args.learning_rate:
        config.learning_rate = args.learning_rate
        
    print(f"Model: {config.model_name}")
    print(f"Architecture: {config.model_type}")
    print(f"GRU units: {config.gru_units}")
    print(f"Lookback: {config.lookback_steps} steps")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Dropout: {config.dropout_rate}")
    
    # 2. Create tracking configuration
    print(f"\n{'='*70}")
    print("INITIALIZING EXPERIMENT TRACKING")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Vertex AI requires run names to match [a-z0-9][a-z0-9-]{0,127}
    # We replace underscores with hyphens to ensure validity
    safe_model_name = config.model_name.replace('_', '-')
    
    # Check for RUN_NAME env var or Arg
    if args.run_name:
        run_name = args.run_name
        # Start Clean: If the user provides a run name, we assume they want THAT run name.
        # But if it looks static (from pipeline definition default), we might want to append timestamp?
        # The pipeline usually sends a unique run_name if constructed with dsl.PIPELINE_JOB_ID_PLACEHOLDER or similar?
        # For now, trust the arg.
        print(f"Using provided run name: {run_name}")
    elif os.environ.get("RUN_NAME"):
        run_name = os.environ.get("RUN_NAME")
        print(f"Using environment run name: {run_name}")
    else:
        run_name = f"{safe_model_name}-{timestamp}"
        print(f"Generated run name: {run_name}")
    
    # Determine log directory (prefer GCS path if provided)
    log_dir = None
    if args.tensorboard_dir:
        # Re-enable subdirectory structure for correct Run Name in TensorBoard
        # The uploader watches the root, so creating a subdir here defines the Run Name in the UI.
        log_dir = f"{args.tensorboard_dir}/{run_name}"
        print(f"Logging to TensorBoard dir: {log_dir}")
    
    tracking_config = TrackingConfig.create_from_model_config(
        model_config=config,
        experiment_name=config.experiment_name or "headway-prediction",
        run_name=run_name,
        log_dir=log_dir,
        vertex_project=config.bq_project,
        vertex_location=config.vertex_location,
        use_vertex_experiments=config.use_vertex_experiments,
        tensorboard_resource_name=args.tensorboard_resource_name,
        histograms=config.track_histograms,
        histogram_freq=config.histogram_freq,
        profiling=config.track_profiling,
        profile_batch_range=config.profile_batch_range
    )
    
    # 3. Initialize experiment tracker
    tracker = ExperimentTracker(tracking_config)
    
    # Configure distribution strategy for GPU support
    strategy = tf.distribute.MirroredStrategy()
    print(f"Distribution Strategy: {strategy.num_replicas_in_sync} devices")
    
    try:
        with strategy.scope():
            # 4. Build model
            print(f"\n{'='*70}")
            print("BUILDING MODEL")
            print("="*70)
            
            model_builder = StackedGRUModel(config)
            print(model_builder.get_architecture_summary())
            
            model = model_builder.create()
            model.summary()
            
            # 5. Log model graph
            tracker.log_graph(model)
            
            # 6. Load data
            print(f"\n{'='*70}")
            print(f"LOADING DATA from {args.input_csv}")
            print("="*70)
            
            trainer = Trainer(config)
            trainer.load_data(args.input_csv)
            
            # 7. Train with tracking
            print(f"\n{'='*70}")
            print("TRAINING MODEL")
            print("="*70)
            print(f"Run: {tracking_config.experiment_name}/{run_name}")
            print(f"TensorBoard: {tracking_config.get_tensorboard_command()}")
            
            history = trainer.train(
                model=model,
                callbacks=tracker.keras_callbacks()
            )
        
        # 8. Log final summary
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print("="*70)
        
        final_metrics = {
            key: float(values[-1]) if isinstance(values, list) else float(values)
            for key, values in history.history.items()
        }
        
        summary_text = f"""
# Training Summary

**Run:** {run_name}  
**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Final Metrics
"""
        for metric_name, metric_value in final_metrics.items():
            summary_text += f"- **{metric_name}**: {metric_value:.4f}\n"
        
        tracker.log_text("experiment/final_summary", summary_text, step=config.epochs)
        
        # Log best metrics
        best_val_loss = min(history.history['val_loss'])
        best_epoch = history.history['val_loss'].index(best_val_loss) + 1
        
        print(f"\nBest Validation Loss: {best_val_loss:.4f} (epoch {best_epoch})")
        

        # 9. Save test set for independent evaluation component
        print(f"\n{'='*70}")
        print(f"SAVING TEST SET to {args.test_dataset_path}")
        print("="*70)
        try:
            trainer.save_test_set(args.test_dataset_path)
            print("✓ Test set saved successfully")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to save test set: {str(e)}")
            # We must raise this to fail the component, otherwise Eval will fail mysteriously later
            raise e

        # 10. Quick in-process evaluation (logging only)
        print(f"\n{'='*70}")
        print("LOGGING METRICS")
        print("="*70)
        
        try:
            _, _, test_dataset = trainer.create_datasets()
            test_results = model.evaluate(test_dataset, verbose=1, return_dict=True)
            
            for metric_name, metric_value in test_results.items():
                tracker.log_metric(f"test/{metric_name}", metric_value, step=config.epochs)
            print("✓ Metrics logged successfully")
        except Exception as e:
            print(f"ERROR: Failed to evaluate model: {str(e)}")
            test_results = {}
        
        # 11. Save model
        print(f"\n{'='*70}")
        print(f"SAVING MODEL to {args.model_dir}")
        print("="*70)
        try:
            # Save as SavedModel (default when path is directory)
            # This is required for the Evaluation component which uses tf.keras.models.load_model(dir)
            
            # Ensure directory exists
            os.makedirs(args.model_dir, exist_ok=True)
            
            print(f"Saving to SavedModel format: {args.model_dir}")
            # Use export if available (Keras 3/recent), else save
            if hasattr(model, 'export'):
                model.export(args.model_dir)
                print("✓ Model exported successfully (SavedModel)")
                
                # Also save .keras for backup/compatibility
                backup_path = os.path.join(args.model_dir, 'model.keras')
                model.save(backup_path)
                print(f"✓ Model saved to .keras format: {backup_path}")
            else:
                model.save(args.model_dir)
                print("✓ Model saved successfully (SavedModel)")
                
        except Exception as e:
            print(f"ERROR: Failed to save model: {str(e)}")
            raise e

        print(f"✓ Experiment: {config.experiment_name}/{run_name}")
        print(f"✓ TensorBoard: {tracking_config.get_tensorboard_command()}")
        
        return model, history, test_results
        
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
