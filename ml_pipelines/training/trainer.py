"""
Trainer Class

Handles model compilation, training loop orchestration, and integration
with experiment tracking.
"""

import os
from typing import List, Optional, Dict, Any
import tensorflow as tf
from ml_pipelines.config import ModelConfig
from ml_pipelines.models.model_builder import compile_model, get_callbacks


class Trainer:
    """
    Orchestrates model training with experiment tracking integration.
    
    Manages:
    - Model compilation with optimizers and loss functions
    - Training loop with callbacks
    - Integration with ExperimentTracker for logging
    - Checkpoint management
    - Learning rate scheduling
    
    Example:
        # Create trainer
        trainer = Trainer(
            model=model,
            config=model_config,
            tracker=experiment_tracker
        )
        
        # Compile model
        trainer.compile(metrics=[rmse_seconds, r_squared])
        
        # Train
        history = trainer.fit(
            train_dataset=train_ds,
            val_dataset=val_ds
        )
        
        # Evaluate
        test_metrics = trainer.evaluate(test_ds)
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        config: ModelConfig,
        tracker: Optional[Any] = None,
        checkpoint_dir: str = "checkpoints",
        steps_per_epoch: Optional[int] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Keras model to train (compiled or uncompiled)
            config: ModelConfig with training parameters
            tracker: Optional ExperimentTracker for logging
            checkpoint_dir: Directory for model checkpoints
            steps_per_epoch: Number of training steps per epoch (for LR scheduling)
        """
        self.model = model
        self.config = config
        self.tracker = tracker
        self.checkpoint_dir = checkpoint_dir
        self.steps_per_epoch = steps_per_epoch
        self.history = None
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"✓ Trainer initialized")
        print(f"  Model: {model.name}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
    
    # =========================================================================
    # Compilation
    # =========================================================================
    
    def compile(
        self,
        metrics: Optional[List] = None,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss: Optional[Any] = None
    ):
        """
        Compile the model for training.
        
        Args:
            metrics: List of metrics to track during training
            optimizer: Custom optimizer (uses config default if None)
            loss: Custom loss function (uses config default if None)
        """
        if optimizer or loss:
            # Custom compilation
            self.model.compile(
                optimizer=optimizer or compile_model.__wrapped__(self.config, self.steps_per_epoch),
                loss=loss or self.config.loss,
                metrics=metrics or []
            )
        else:
            # Standard compilation from config
            compile_model(
                self.model,
                self.config,
                metrics=metrics,
                steps_per_epoch=self.steps_per_epoch
            )
        
        print(f"✓ Model compiled successfully")
    
    # =========================================================================
    # Training
    # =========================================================================
    
    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: Optional[int] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1,
        **fit_kwargs
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            train_dataset: Training tf.data.Dataset
            val_dataset: Validation tf.data.Dataset (optional)
            epochs: Number of epochs (uses config default if None)
            callbacks: Additional callbacks (tracker callbacks added automatically)
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
            **fit_kwargs: Additional arguments for model.fit()
            
        Returns:
            Keras History object
        """
        epochs = epochs or self.config.epochs
        
        # Collect all callbacks
        all_callbacks = self._prepare_callbacks(callbacks)
        
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"{'='*80}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Callbacks: {len(all_callbacks)} total")
        if self.tracker:
            print(f"Experiment tracking: ENABLED")
            print(f"  Experiment: {self.tracker.config.experiment_name}")
            print(f"  Run: {self.tracker.config.run_name}")
        print(f"{'='*80}\n")
        
        # Train model
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=all_callbacks,
            verbose=verbose,
            **fit_kwargs
        )
        
        print(f"\n{'='*80}")
        print(f"Training Complete")
        print(f"{'='*80}")
        self._print_training_summary()
        print(f"{'='*80}\n")
        
        return self.history
    
    def _prepare_callbacks(
        self,
        custom_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> List[tf.keras.callbacks.Callback]:
        """
        Prepare all callbacks for training.
        
        Args:
            custom_callbacks: User-provided callbacks
            
        Returns:
            List of all callbacks to use
        """
        callbacks = []
        
        # Add standard training callbacks
        callbacks.extend(get_callbacks(
            self.config,
            checkpoint_dir=self.checkpoint_dir,
            monitor="val_loss" if "val" in self.model.metrics_names else "loss",
            mode="min"
        ))
        
        # Add tracker callbacks if tracker is provided
        if self.tracker:
            callbacks.extend(self.tracker.keras_callbacks())
        
        # Add custom callbacks
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        
        return callbacks
    
    def _print_training_summary(self):
        """Print summary of training results."""
        if self.history is None:
            return
        
        history_dict = self.history.history
        
        # Find best epoch
        val_loss_key = "val_loss" if "val_loss" in history_dict else "loss"
        best_epoch = int(np.argmin(history_dict[val_loss_key])) + 1
        best_val_loss = min(history_dict[val_loss_key])
        
        print(f"Best epoch: {best_epoch}/{len(history_dict['loss'])}")
        print(f"Best val_loss: {best_val_loss:.6f}")
        
        # Print final metrics
        print(f"\nFinal metrics:")
        for metric_name, values in history_dict.items():
            if values:
                final_value = values[-1]
                print(f"  {metric_name}: {final_value:.6f}")
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    def evaluate(
        self,
        test_dataset: tf.data.Dataset,
        verbose: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_dataset: Test tf.data.Dataset
            verbose: Verbosity mode
            
        Returns:
            Dictionary of metric names to values
        """
        print(f"\n{'='*80}")
        print(f"Evaluating Model on Test Set")
        print(f"{'='*80}\n")
        
        results = self.model.evaluate(test_dataset, verbose=verbose, return_dict=True)
        
        print(f"\n{'='*80}")
        print(f"Test Results:")
        for metric_name, value in results.items():
            print(f"  {metric_name}: {value:.6f}")
        print(f"{'='*80}\n")
        
        # Log to tracker if available
        if self.tracker:
            self.tracker.log_scalars(results, step=0, prefix="test")
        
        return results
    
    # =========================================================================
    # Model Management
    # =========================================================================
    
    def save_model(
        self,
        save_path: str,
        save_format: str = "keras"
    ):
        """
        Save trained model.
        
        Args:
            save_path: Path to save model
            save_format: Format ('keras', 'tf', 'h5')
        """
        self.model.save(save_path, save_format=save_format)
        print(f"✓ Model saved to: {save_path}")
    
    def load_weights(self, weights_path: str):
        """
        Load model weights.
        
        Args:
            weights_path: Path to weights file
        """
        self.model.load_weights(weights_path)
        print(f"✓ Weights loaded from: {weights_path}")
    
    def get_best_checkpoint_path(self) -> str:
        """
        Get path to best model checkpoint.
        
        Returns:
            Path to best model checkpoint
        """
        return os.path.join(self.checkpoint_dir, "best_model.keras")
    
    def load_best_checkpoint(self):
        """Load the best checkpoint from training."""
        checkpoint_path = self.get_best_checkpoint_path()
        if os.path.exists(checkpoint_path):
            self.model = tf.keras.models.load_model(checkpoint_path)
            print(f"✓ Loaded best checkpoint from: {checkpoint_path}")
        else:
            print(f"⚠ Warning: No checkpoint found at {checkpoint_path}")


# Import numpy for training summary
import numpy as np
