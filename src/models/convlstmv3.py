"""
ConvLSTM v3 for Headway Prediction

4-layer encoder-decoder: 2 encoder + 2 decoder ConvLSTM layers.
No normalization layers. Clean state transfer.

Usage:
    from src.models.convlstmv3 import ConvLSTM
    
    model = ConvLSTM(config).build()
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


class ConvLSTM:
    """
    4-layer ConvLSTM encoder-decoder for spatiotemporal headway prediction.
    
    Architecture:
        Encoder: 2 ConvLSTM layers on 30-step history → (h, c) state
        Decoder: 2 ConvLSTM layers on 15-step schedule, initialized with encoder state
        Output: Conv3D projection → (15, 66, 2)
    
    Args:
        config: Config object with hyperparameters
    """
    
    custom_objects = {"BroadcastLayer": BroadcastLayer}
    
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
        
        # === ENCODER: 2 layers, extract final state ===
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
        sched = BroadcastLayer(self.stations, self.directions, name="broadcast_schedule")(
            layers.Reshape((self.horizon, 1, 1, 2))(schedule_in)
        )
        # sched shape: (B, 15, 66, 2, 2)
        
        # Project to filter count
        sched = layers.Conv3D(
            filters=self.filters,
            kernel_size=(1, 1, 1),
            padding="same",
            name="schedule_projection"
        )(sched)
        # sched shape: (B, 15, 66, 2, 32)
        
        # === DECODER: 2 layers, initialized with encoder state ===
        x = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel,
            padding="same",
            return_sequences=True,
            activation="relu",
            unroll=True,
            name="decoder_1"
        )(sched, initial_state=[state_h, state_c])
        
        x = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel,
            padding="same",
            return_sequences=True,
            activation="relu",
            unroll=True,
            name="decoder_2"
        )(x)
        # x shape: (B, 15, S, D, filters)
        
        # === OUTPUT: Project to single channel ===
        x = layers.Conv3D(
            filters=1,
            kernel_size=(1, 1, 1),
            activation="linear",
            dtype="float32",
            name="output_conv"
        )(x)
        
        # Remove channel dim
        output = layers.Reshape(
            (self.horizon, self.stations, self.directions),
            name="output"
        )(x)
        
        model = keras.Model(
            inputs=[headway_in, schedule_in],
            outputs=output,
            name="ConvLSTM"
        )
        
        return model
    
    @classmethod
    def load(cls, path: str) -> keras.Model:
        """Load a saved model with custom objects."""
        return keras.models.load_model(path, custom_objects=cls.custom_objects)
