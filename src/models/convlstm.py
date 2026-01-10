"""
ConvLSTM Encoder-Decoder for Headway Prediction

Architecture based on: Usama & Koutsopoulos (2025) arXiv:2510.03121
Implementation follows "Strategy 3: Early Fusion via Spatial Broadcasting"

Class-based architecture for clean encapsulation and easy subclassing.

Usage:
    from src.models.convlstm import ConvLSTM
    
    model_builder = ConvLSTM(config)
    model = model_builder.build_model()
    model.summary()
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.config import Config


class BroadcastScheduleLayer(layers.Layer):
    """
    Broadcasts terminal schedule to all stations.
    
    Input: (Batch, Time, 2) - schedule for 2 terminals
    Output: (Batch, Time, Stations, 2) - schedule broadcast to all stations
    
    Logic:
        - Terminal 0 schedule → All stations, Direction 0
        - Terminal 1 schedule → All stations, Direction 1
    """
    
    def __init__(self, num_stations, **kwargs):
        super().__init__(**kwargs)
        self.num_stations = num_stations
    
    def call(self, inputs):
        # inputs shape: (Batch, Time, 2)
        # We want: (Batch, Time, Stations, 2)
        
        # Expand dims: (B, T, 2) -> (B, T, 1, 2)
        expanded = tf.expand_dims(inputs, axis=2)
        
        # Tile along station axis: (B, T, 1, 2) -> (B, T, Stations, 2)
        broadcasted = tf.tile(expanded, [1, 1, self.num_stations, 1])
        
        return broadcasted
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_stations": self.num_stations})
        return config


class BroadcastTemporalLayer(layers.Layer):
    """
    Broadcasts temporal features to all stations and directions.
    
    Input: (Batch, Time, 4) - cyclical time features (hour_sin, hour_cos, day_sin, day_cos)
    Output: (Batch, Time, Stations, Directions, 4) - features broadcast spatially
    
    This allows the model to know "what time is it" at each spatial location.
    """
    
    def __init__(self, num_stations, num_directions=2, **kwargs):
        super().__init__(**kwargs)
        self.num_stations = num_stations
        self.num_directions = num_directions
    
    def call(self, inputs):
        # inputs shape: (Batch, Time, 4)
        # We want: (Batch, Time, Stations, Directions, 4)
        
        # Expand dims: (B, T, 4) -> (B, T, 1, 1, 4)
        expanded = tf.expand_dims(tf.expand_dims(inputs, axis=2), axis=3)
        
        # Tile: (B, T, 1, 1, 4) -> (B, T, Stations, Directions, 4)
        broadcasted = tf.tile(expanded, [1, 1, self.num_stations, self.num_directions, 1])
        
        return broadcasted
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_stations": self.num_stations,
            "num_directions": self.num_directions
        })
        return config


class ConvLSTM:
    """
    ConvLSTM Encoder-Decoder with Terminal Schedule Broadcasting.
    
    Encapsulates:
        - Model architecture configuration
        - Custom layers (BroadcastScheduleLayer)
        - Build logic for encoder-decoder structure
        - State transfer between encoder and decoder
    
    Attributes:
        config: Configuration object with hyperparameters
        model: Built Keras model (after calling build_model())
    
    Example:
        builder = ConvLSTM(config)
        model = builder.build_model()
        model.compile(optimizer='adam', loss='mse')
        model.fit(train_ds, validation_data=val_ds)
    """
    
    # Class-level registry of custom layers for model loading
    custom_objects = {
        'BroadcastScheduleLayer': BroadcastScheduleLayer,
        'BroadcastTemporalLayer': BroadcastTemporalLayer
    }
    
    def __init__(self, config: Config = None, use_temporal: bool = False):
        """
        Initialize the ConvLSTM builder.
        
        Args:
            config: Configuration object with hyperparameters.
                   If None, uses default Config().
            use_temporal: If True, adds temporal_input for cyclical time features
        """
        self.config = config if config is not None else Config()
        self.model = None
        self.use_temporal = use_temporal
        
        # Extract dimensions from config
        self.lookback = self.config.LOOKBACK_MINS       # 30
        self.forecast = self.config.FORECAST_MINS       # 15
        self.num_stations = self.config.NUM_STATIONS    # 66
        self.num_directions = 2                         # N and S
        self.num_terminals = 2                          # North and South terminals
        self.filters = self.config.FILTERS              # 32
        self.kernel_size = self.config.KERNEL_SIZE      # (3, 3)
        self.temporal_features = 4                      # hour_sin, hour_cos, day_sin, day_cos
    
    def build_model(self) -> keras.Model:
        """
        Build the ConvLSTM Encoder-Decoder model.
        
        Returns:
            Keras Functional API model with inputs:
                - headway_input: (B, 30, 66, 2)
                - schedule_input: (B, 15, 2)
                - temporal_input (optional): (B, 30, 4)
            Output: (B, 15, 66, 2)
        """
        # =================================================================
        # Input Definitions
        # =================================================================
        input_headway = keras.Input(
            shape=(self.lookback, self.num_stations, self.num_directions),
            name="headway_input"
        )
        
        input_schedule = keras.Input(
            shape=(self.forecast, self.num_terminals),
            name="schedule_input"
        )
        
        # Optional temporal input
        input_temporal = None
        if self.use_temporal:
            input_temporal = keras.Input(
                shape=(self.lookback, self.temporal_features),
                name="temporal_input"
            )
        
        # =================================================================
        # Schedule Broadcasting
        # =================================================================
        schedule_broadcast = BroadcastScheduleLayer(
            num_stations=self.num_stations,
            name="broadcast_schedule"
        )(input_schedule)
        
        # =================================================================
        # Reshape for ConvLSTM2D (5D input required)
        # =================================================================
        # Headway: (B, T, 66, 2) -> (B, T, 66, 2, 1)
        headway_5d = layers.Reshape(
            (self.lookback, self.num_stations, self.num_directions, 1),
            name="reshape_headway_5d"
        )(input_headway)
        
        # =================================================================
        # Temporal Feature Fusion (Early Fusion)
        # =================================================================
        if self.use_temporal:
            # Broadcast temporal to (B, T, 66, 2, 4)
            temporal_broadcast = BroadcastTemporalLayer(
                num_stations=self.num_stations,
                num_directions=self.num_directions,
                name="broadcast_temporal"
            )(input_temporal)
            
            # Concatenate: headway (1 ch) + temporal (4 ch) = 5 channels
            encoder_input = layers.Concatenate(axis=-1, name="fuse_temporal")(
                [headway_5d, temporal_broadcast]
            )
        else:
            encoder_input = headway_5d
        
        schedule_5d = layers.Reshape(
            (self.forecast, self.num_stations, self.num_directions, 1),
            name="reshape_schedule_5d"
        )(schedule_broadcast)
        
        # =================================================================
        # Encoder (uses fused encoder_input with temporal features)
        # =================================================================
        encoder_x = self._build_encoder(encoder_input)
        state_h, state_c = encoder_x[1], encoder_x[2]
        
        # =================================================================
        # Decoder
        # =================================================================
        decoder_x = self._build_decoder(schedule_5d, state_h, state_c)
        
        # =================================================================
        # Output Projection
        # =================================================================
        output = self._build_output(decoder_x)
        
        # =================================================================
        # Build Model
        # =================================================================
        inputs_dict = {
            "headway_input": input_headway,
            "schedule_input": input_schedule
        }
        if self.use_temporal:
            inputs_dict["temporal_input"] = input_temporal
        
        self.model = keras.Model(
            inputs=inputs_dict,
            outputs=output,
            name="ConvLSTM_HeadwayPredictor"
        )
        
        return self.model
    
    def _build_encoder(self, x):
        """
        Build encoder layers.
        
        Args:
            x: Input tensor (B, T, H, W, C)
        
        Returns:
            Tuple of (encoded_sequence, state_h, state_c)
        """
        # Encoder Layer 1
        x = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same",
            return_sequences=True,
            activation="relu",
            unroll=True,  # Static graph: trades memory for speed
            name="encoder_convlstm_1"
        )(x)
        
        # GroupNorm: batch-independent normalization, stable for RNNs
        # groups=32 matches filter count (32 filters = 32 groups = per-channel norm)
        x = layers.GroupNormalization(groups=32, name="encoder_gn_1")(x)
        
        # Encoder Layer 2 - Returns states for decoder
        x, state_h, state_c = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same",
            return_sequences=True,
            return_state=True,
            activation="relu",
            unroll=True,  # Static graph: trades memory for speed
            name="encoder_convlstm_2"
        )(x)
        
        x = layers.GroupNormalization(groups=32, name="encoder_gn_2")(x)
        
        return x, state_h, state_c
    
    def _build_decoder(self, x, state_h, state_c):
        """
        Build decoder layers initialized with encoder states.
        
        Args:
            x: Input tensor (schedule_5d)
            state_h: Encoder hidden state
            state_c: Encoder cell state
        
        Returns:
            Decoded tensor
        """
        # Decoder Layer 1 - Initialized with encoder states
        x = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same",
            return_sequences=True,
            activation="relu",
            unroll=True,  # Static graph: trades memory for speed
            name="decoder_convlstm_1"
        )(x, initial_state=[state_h, state_c])
        
        x = layers.GroupNormalization(groups=32, name="decoder_gn_1")(x)
        
        # Decoder Layer 2
        x, _, _ = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same",
            return_sequences=True,
            return_state=True,
            activation="relu",
            unroll=True,  # Static graph: trades memory for speed
            name="decoder_convlstm_2"
        )(x)
        
        x = layers.GroupNormalization(groups=32, name="decoder_gn_2")(x)
        
        return x
    
    def _build_output(self, x):
        """
        Build output projection layers.
        
        Args:
            x: Decoder output tensor
        
        Returns:
            Output tensor (B, forecast, stations, directions)
        """
        # Conv3D projection: (B, T, 66, 2, Filters) -> (B, T, 66, 2, 1)
        x = layers.Conv3D(
            filters=1,
            kernel_size=(3, 3, 1),
            padding="same",
            activation="relu",
            name="output_projection"
        )(x)
        
        # Reshape: (B, T, 66, 2, 1) -> (B, T, 66, 2)
        output = layers.Reshape(
            (self.forecast, self.num_stations, self.num_directions),
            name="output_reshape"
        )(x)
        
        return output
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build_model()
        self.model.summary()
    
    def get_config(self) -> dict:
        """Return model configuration for serialization."""
        return {
            "lookback": self.lookback,
            "forecast": self.forecast,
            "num_stations": self.num_stations,
            "num_directions": self.num_directions,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "use_temporal": self.use_temporal,
        }
    
    @classmethod
    def load_model(cls, path: str) -> keras.Model:
        """
        Load a saved model with custom objects registered.
        
        Args:
            path: Path to saved .keras file
        
        Returns:
            Loaded Keras model
        """
        return keras.models.load_model(path, custom_objects=cls.custom_objects)


# =============================================================================
# Convenience Function (backwards compatibility)
# =============================================================================

def build_convlstm_model(config: Config = None) -> keras.Model:
    """
    Build ConvLSTM model (function wrapper for backwards compatibility).
    
    Args:
        config: Configuration object
    
    Returns:
        Keras model
    """
    return ConvLSTM(config).build_model()


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Building ConvLSTM Encoder-Decoder Model...")
    print("=" * 60)
    
    builder = ConvLSTM()
    model = builder.build_model()
    builder.summary()
    
    print("\n" + "=" * 60)
    print(f"Total parameters: {model.count_params():,}")
    
    print("\n" + "=" * 60)
    print("INPUT/OUTPUT SHAPE VERIFICATION")
    print("=" * 60)
    print("Model Inputs:")
    for name, spec in model.input.items():
        print(f"  {name}: {spec.shape}")
    print(f"Model Output: {model.output.shape}")
    
    print("\nModel Config:")
    print(builder.get_config())
