"""
Simple 2-Layer ConvLSTM for Headway Prediction

A minimal architecture: stack 2 ConvLSTM layers, project to output.
No encoder-decoder complexity. Keeps spatial structure throughout.

Usage:
    from src.models.simple_convlstm import SimpleConvLSTM
    
    model = SimpleConvLSTM(config).build()
    model.summary()
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.config import Config


class BroadcastLayer(layers.Layer):
    """Broadcasts (B, T, 1, 1, F) → (B, T, S, D, F) via tile."""
    
    def __init__(self, stations, directions, **kwargs):
        super().__init__(**kwargs)
        self.stations = stations
        self.directions = directions
    
    def call(self, inputs):
        return tf.tile(inputs, [1, 1, self.stations, self.directions, 1])
    
    def get_config(self):
        config = super().get_config()
        config.update({"stations": self.stations, "directions": self.directions})
        return config


class SimpleConvLSTM:
    """
    Simple 2-layer ConvLSTM for spatiotemporal headway prediction.
    
    Architecture:
        Input (30, S, D, 1) → ConvLSTM → ConvLSTM → RepeatVector → Conv3D → Output (15, S, D)
    
    Key: Never flatten. Use Conv3D for output projection.
    
    Args:
        config: Config object with hyperparameters
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.lookback = config.LOOKBACK_MINS
        self.horizon = config.FORECAST_MINS
        self.stations = config.NUM_STATIONS
        self.directions = 2
        self.filters = config.FILTERS
        self.kernel = config.KERNEL_SIZE
    
    def build(self) -> keras.Model:
        """Build and return the model."""
        
        # Inputs
        headway_in = layers.Input(
            shape=(self.lookback, self.stations, self.directions),
            name="headway_input"
        )
        schedule_in = layers.Input(
            shape=(self.horizon, 2),
            name="schedule_input"
        )
        
        # Add channel dimension: (B, T, S, D) → (B, T, S, D, 1)
        x = layers.Reshape(
            (self.lookback, self.stations, self.directions, 1)
        )(headway_in)
        
        # === ENCODER: Process history, extract state ===
        x = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel,
            padding="same",
            return_sequences=True,
            activation="relu",
            unroll=True,
            name="encoder_1"
        )(x)
        
        x, state_h, state_c = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel,
            padding="same",
            return_sequences=False,
            return_state=True,
            activation="relu",
            unroll=True,
            name="encoder_2"
        )(x)
        # state_h, state_c: (B, S, D, filters)
        
        # === DECODER INPUT: Broadcast schedule to spatial dims ===
        # schedule_in: (B, 15, 2) → (B, 15, S, D, 1)
        sched = BroadcastLayer(self.stations, self.directions, name="broadcast_schedule")(
            layers.Reshape((self.horizon, 1, 1, 2))(schedule_in)
        )
        # sched shape: (B, 15, 66, 2, 2)
        
        # Pad to match filter count for ConvLSTM input
        sched = layers.Conv3D(
            filters=self.filters,
            kernel_size=(1, 1, 1),
            padding="same",
            name="schedule_projection"
        )(sched)
        # sched shape: (B, 15, 66, 2, 32)
        
        # === DECODER: Generate forecast using encoder state ===
        x = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel,
            padding="same",
            return_sequences=True,
            activation="relu",
            unroll=True,
            name="decoder"
        )(sched, initial_state=[state_h, state_c])
        # x shape: (B, 15, S, D, filters)
        
        # === OUTPUT: Project to single channel ===
        x = layers.Conv3D(
            filters=1,
            kernel_size=(1, 1, 1),
            activation="linear",
            dtype="float32",
            name="output_conv"
        )(x)
        # x shape: (B, horizon, S, D, 1)
        
        # Remove channel dim
        output = layers.Reshape(
            (self.horizon, self.stations, self.directions),
            name="output"
        )(x)
        
        model = keras.Model(
            inputs=[headway_in, schedule_in],
            outputs=output,
            name="SimpleConvLSTM"
        )
        
        return model
    
    @staticmethod
    def load(path: str) -> keras.Model:
        """Load a saved model."""
        return keras.models.load_model(path)
