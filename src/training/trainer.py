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
            warmup_epochs = 5
            warmup_steps = self.steps_per_epoch * warmup_epochs
            
            # CosineDecay with Warmup: ramp LR from near-zero to target over first 5 epochs
            # This stabilizes early training before entering high-curvature regions
            lr_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-6,              # Start near zero
                decay_steps=total_steps - warmup_steps,  # Decay phase
                alpha=0.01,                              # Final LR = 1% of peak
                warmup_target=self.config.LEARNING_RATE, # Ramp up to target
                warmup_steps=warmup_steps                # Over 5 epochs
            )
            # AdamW with tuned hyperparameters for bfloat16 stability:
            # - beta_2=0.95: Faster curvature tracking (~20 steps vs 1000) to avoid Edge of Stability spikes
            # - epsilon=1e-6: Safe buffer above bfloat16 noise floor (was 1e-7)
            # - weight_decay=0.01: Decoupled from gradient, keeps moment estimates clean
            optimizer = keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.95,
                epsilon=1e-6,
                clipnorm=1.0
            )
            print(f"Using AdamW with CosineDecay + Warmup:")
            print(f"  Warmup: 1e-6 → {self.config.LEARNING_RATE} over {warmup_steps} steps ({warmup_epochs} epochs)")
            print(f"  Decay:  {self.config.LEARNING_RATE} → {self.config.LEARNING_RATE * 0.01} over {total_steps - warmup_steps} steps")
            print(f"  AdamW: beta_2=0.95, epsilon=1e-6, weight_decay=0.01")
            print("Gradient clipping: clipnorm=1.0")
        else:
            # Fallback to constant LR if steps_per_epoch not provided
            optimizer = keras.optimizers.AdamW(
                learning_rate=self.config.LEARNING_RATE,
                weight_decay=0.01,
                beta_2=0.95,
                epsilon=1e-6,
                clipnorm=1.0
            )
            print(f"Using AdamW with constant LR: {self.config.LEARNING_RATE} (pass steps_per_epoch for CosineDecay)")
            print(f"  AdamW: beta_2=0.95, epsilon=1e-6, weight_decay=0.01")
            print("Gradient clipping: clipnorm=1.0")

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
        # Note: ReduceLROnPlateau is NOT included because we use CosineDecay schedule.
        # These are mutually exclusive — CosineDecay controls LR, callbacks can't override it.
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
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