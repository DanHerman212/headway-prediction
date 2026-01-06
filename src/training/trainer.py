# this class handles the compilation (optimizer/loss) and training loop with callbacks
import os
import tensorflow as tf
from tensorflow import keras
from src.config import Config
from src.metrics import rmse_seconds, r_squared

class Trainer:
    def __init__(self, model, config: Config, checkpoint_dir: str = "models"):
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir

    def compile_model(self):
        """Configures model for training with production-relevant metrics."""
        optimizer = keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)

        # Loss: MSE (penalizes large outliers/delays heavily)
        # Metrics: 
        #   - rmse_seconds: RMSE in real units (seconds) for interpretability
        #   - r_squared: Coefficient of determination (0-1, higher is better)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[rmse_seconds, r_squared]
        )

    def fit(self, train_dataset, val_dataset, patience=5, reduce_lr_patience=3):
        """
        Runs training loop with callbacks.
        
        Args:
            train_dataset: Training tf.data.Dataset
            val_dataset: Validation tf.data.Dataset
            patience: Early stopping patience (default 5, paper uses 50)
            reduce_lr_patience: ReduceLROnPlateau patience (default 3)
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.keras")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                verbose=0
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0
            )
        ]

        print(f"Starting training for {self.config.EPOCHS} epochs...")
        print(f"  Early stopping patience: {patience}")
        print(f"  Checkpoint path: {checkpoint_path}")

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.EPOCHS,
            callbacks=callbacks
        )
        
        return history