"""Model training for headway prediction."""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config.model_config import ModelConfig

from evaluation.metrics import MAESeconds


class Trainer:
    """Handles model training with time series data."""
    
    def __init__(self, config: "ModelConfig"):
        """
        Initialize trainer.
        
        Args:
            config: ModelConfig instance with training parameters
        """
        self.config = config
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
        data = pd.read_csv(data_path)
        
        # Create input arrays
        self.input_x = data.values  # All features (51751, 8)
        self.input_t = data['log_headway'].values  # Target headway (51751,)
        self.input_r = data[['route_A', 'route_C', 'route_E']].values  # Target route (51751, 3)
    
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
        train_end = int(n * self.config.train_split)
        val_end = int(n * (self.config.train_split + self.config.val_split))
        
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
        classification_loss = keras.losses.SparseCategoricalCrossentropy()
        
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
                filepath=f"{self.config.checkpoint_dir}/best_model.keras",
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
        
    print(f"Model: {config.model_name}")
    print(f"Architecture: {config.model_type}")
    print(f"GRU units: {config.gru_units}")
    print(f"Lookback: {config.lookback_steps} steps")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    
    # 2. Create tracking configuration
    print(f"\n{'='*70}")
    print("INITIALIZING EXPERIMENT TRACKING")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config.model_name}-{timestamp}"
    
    tracking_config = TrackingConfig.create_from_model_config(
        model_config=config,
        experiment_name=config.experiment_name or "headway-prediction",
        run_name=run_name,
        vertex_project=config.bq_project,
        vertex_location=config.vertex_location,
        use_vertex_experiments=config.use_vertex_experiments,
        histograms=config.track_histograms,
        histogram_freq=config.histogram_freq,
        profiling=config.track_profiling,
        profile_batch_range=config.profile_batch_range
    )
    
    # 3. Initialize experiment tracker
    tracker = ExperimentTracker(tracking_config)
    
    try:
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
        
        # 9. Evaluate on test set (within training process for immediate feedback)
        # Note: We also save test dataset for the standalone component
        print(f"\n{'='*70}")
        print("EVALUATING ON TEST SET")
        print("="*70)
        
        _, _, test_dataset = trainer.create_datasets()
        # Note: create_datasets() uses internal state. 
        # We need to save the test *dataframe* corresponding to these indices for the next component.
        
        # Logic to save test split matches create_datasets slice:
        # test_dataset = build_from_indices(val_end, None, is_training=False)
        # val_end = int(n * (train_split + val_split))
        
        # Re-read raw data to slice it (trainer.input_x is numpy array)
        # We can use trainer.input_x directly but we need it as a dataframe or CSV to save?
        # The next component expects a CSV.
        
        df_full = pd.read_csv(args.input_csv)
        n = len(df_full)
        val_end = int(n * (config.train_split + config.val_split))
        df_test = df_full.iloc[val_end:]
        
        os.makedirs(os.path.dirname(args.test_dataset_path), exist_ok=True)
        print(f"Saving test dataset ({len(df_test)} rows) to {args.test_dataset_path}")
        df_test.to_csv(args.test_dataset_path, index=False)
        
        
        test_results = model.evaluate(test_dataset, verbose=1, return_dict=True)
        
        print("\nTest Set Results:")
        for metric_name, metric_value in test_results.items():
            print(f"  {metric_name}: {metric_value:.4f}")
            tracker.log_metric(f"test/{metric_name}", metric_value, step=config.epochs)
        
        # 10. Save model
        print(f"Saving model to {args.model_dir}")
        model.save(args.model_dir)
        print(f"\n✓ Model saved: {args.model_dir}")
        print(f"✓ Experiment: {config.experiment_name}/{run_name}")
        print(f"✓ TensorBoard: {tracking_config.get_tensorboard_command()}")
        
        return model, history, test_results
        
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
