# this class handles the compilation (optimizer/loss) and training loop with callbacks
import os
import tensorflow as tf
from tensorflow import keras
from src.config import Config

class Trainer:
    def __init__(self, model, config: Config, checkpoint_dir: str = "models"):
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir

    def compile_model(self):
        """configures model for training"""
        optimizer = keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)

        # We use MSE (mean squared error) for the loss because we want to
        # penalize large outliers (major delays) heavily
        # we track MAE (mean absolute error) because it's easier to read ("off by X minutes")
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
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
                verbose=1
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