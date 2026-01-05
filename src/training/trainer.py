# this class handles the compilation (optimizer/loss) and training loop with callbacks
import tensorflow as tf
from tensorflow import keras
from src.config import Config

class Trainer:
    def __init__(self, model, config: Config):
        self.model = model
        self.config = config

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

    def fit(self, train_dataset, val_dataset):
        """runs training loop with callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
            filepath="best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
            )
        ]

        print(f"Starting training for {self.config.EPOCHS} epochs...")

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.EPOCHS,
            callbacks=callbacks
        )
        
        return history