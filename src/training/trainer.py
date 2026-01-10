# This class handles the compilation (optimizer/loss) and training loop with callbacks
import os
from typing import List, Optional
import tensorflow as tf
from tensorflow import keras
from src.config import Config
from src.metrics import rmse_seconds, r_squared


class Trainer:
    """
    Handles model compilation and training with optional tracking integration.
    
    Follows the established architecture:
        Config -> Model -> Trainer -> Evaluator
    
    Tracking integration:
        Pass callbacks from src/tracking/ to enable TensorBoard logging.
    """
    
    def __init__(self, model, config: Config, checkpoint_dir: str = "models", steps_per_epoch: int = None):
        """
        Args:
            model: Compiled or uncompiled Keras model
            config: Configuration object with hyperparameters
            checkpoint_dir: Directory for model checkpoints
            steps_per_epoch: Number of training steps per epoch (for LR scheduling).
                            If None, uses estimate based on typical dataset size.
        """
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.steps_per_epoch = steps_per_epoch

    def compile_model(self):
        """Configures model for training with production-relevant metrics."""
        
        # Cosine Decay Learning Rate Schedule
        # Smoothly decays LR from initial value to near-zero over training
        # More stable than step decay for recurrent networks
        if self.steps_per_epoch is not None:
            total_steps = self.steps_per_epoch * self.config.EPOCHS
            lr_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.LEARNING_RATE,
                decay_steps=total_steps,
                alpha=0.01  # Final LR = 1% of initial (not full zero)
            )
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            print(f"Using CosineDecay: {self.config.LEARNING_RATE} → {self.config.LEARNING_RATE * 0.01} over {total_steps} steps")
        else:
            # Fallback to constant LR if steps_per_epoch not provided
            optimizer = keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
            print(f"Using constant LR: {self.config.LEARNING_RATE} (pass steps_per_epoch for CosineDecay)")

        # Loss: MSE (penalizes large outliers/delays heavily)
        # Metrics: 
        #   - rmse_seconds: RMSE in real units (seconds) for interpretability
        #   - r_squared: Coefficient of determination (0-1, higher is better)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[rmse_seconds, r_squared],
            jit_compile=True  # XLA: Fuses kernels, reduces launch overhead
        )
        print("XLA JIT compilation enabled")

    def fit(
        self, 
        train_dataset, 
        val_dataset, 
        patience: Optional[int] = None,
        reduce_lr_patience: int = 3,
        extra_callbacks: Optional[List[keras.callbacks.Callback]] = None,
    ):
        """
        Runs training loop with callbacks.
        
        Args:
            train_dataset: Training tf.data.Dataset
            val_dataset: Validation tf.data.Dataset
            patience: Early stopping patience (default: from Config)
            reduce_lr_patience: ReduceLROnPlateau patience
            extra_callbacks: Additional callbacks (e.g., from src/tracking/)
        
        Returns:
            Keras History object
        """
        # Use config default if not specified
        if patience is None:
            patience = self.config.EARLY_STOPPING_PATIENCE
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.keras")
        
        # Core callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Add tracking callbacks if provided
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        print(f"Starting training for {self.config.EPOCHS} epochs...")
        print(f"  Early stopping patience: {patience}")
        print(f"  Checkpoint path: {checkpoint_path}")
        if extra_callbacks:
            print(f"  Extra callbacks: {[type(cb).__name__ for cb in extra_callbacks]}")

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.EPOCHS,
            callbacks=callbacks
        )
        
        return history