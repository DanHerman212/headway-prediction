"""
ConvLSTM Encoder-Decoder with Terminal Schedule Broadcasting for Headway Prediction

Architecture based on: Usama & Koutsopoulos (2025) arXiv:2510.03121
Implementation follows "Strategy 3: Early Fusion via Spatial Broadcasting"

Key Design:
    1. Accept terminal schedule (2 values) and broadcast to all 66 stations
    2. Handle bidirectional train operations (2 directions per station)
    3. ConvLSTM Encoder processes historical headway state
    4. ConvLSTM Decoder generates forecasts conditioned on future schedule
    5. State transfer from encoder to decoder carries spatial memory

Data Shapes:
    - headway_matrix_full.npy: (T, 66, 2, 1) = (Time, Stations, Directions, Channel)
    - schedule_matrix_full.npy: (T, 2, 1) = (Time, Terminals, Channel)

Model Inputs:
    - headway_input: (Batch, Lookback, Stations, Directions) = (B, 30, 66, 2)
    - schedule_input: (Batch, Forecast, Terminals) = (B, 15, 2)

Model Output:
    - (Batch, Forecast, Stations, Directions) = (B, 15, 66, 2)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.config import Config


# =============================================================================
# Custom Layers
# =============================================================================

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


# =============================================================================
# Model Builder
# =============================================================================

def build_convlstm_model(config: Config = None) -> keras.Model:
    """
    Build the ConvLSTM Encoder-Decoder model with schedule broadcasting.
    
    Matches actual data structure:
        - headway: (T, 66, 2, 1) -> input (B, 30, 66, 2)
        - schedule: (T, 2, 1) -> input (B, 15, 2)
    
    Args:
        config: Configuration object with hyperparameters.
                If None, uses default Config().
    
    Returns:
        Keras Functional API model with two inputs.
    """
    if config is None:
        config = Config()
    
    # Extract dimensions from config
    lookback = config.LOOKBACK_MINS       # 30
    forecast = config.FORECAST_MINS       # 15
    num_stations = config.NUM_STATIONS    # 66
    num_directions = 2                    # N and S
    num_terminals = 2                     # North and South terminals
    filters = config.FILTERS              # 32 (per paper Table 1)
    kernel_size = config.KERNEL_SIZE      # (3, 3)
    
    # =========================================================================
    # Input Definitions (Matching Actual Data)
    # =========================================================================
    
    # Input 1: Historical Headway Grid
    # Shape: (Batch, Lookback, Stations, Directions)
    input_headway = keras.Input(
        shape=(lookback, num_stations, num_directions),
        name="headway_input"
    )
    
    # Input 2: Future Terminal Schedule (just 2 values per timestep)
    # Shape: (Batch, Forecast, Terminals)
    input_schedule = keras.Input(
        shape=(forecast, num_terminals),
        name="schedule_input"
    )
    
    # =========================================================================
    # Schedule Broadcasting
    # =========================================================================
    # Expand (B, 15, 2) -> (B, 15, 66, 2)
    # Each terminal's schedule is copied to all 66 stations
    
    schedule_broadcast = BroadcastScheduleLayer(
        num_stations=num_stations,
        name="broadcast_schedule"
    )(input_schedule)
    
    # =========================================================================
    # Reshape for ConvLSTM2D
    # =========================================================================
    # ConvLSTM2D expects 5D: (Batch, Time, Height, Width, Channels)
    # We use: Height=Stations (66), Width=Directions (2), Channels=1
    # This preserves the spatial structure of the subway line
    
    # Reshape headway: (B, T, 66, 2) -> (B, T, 66, 2, 1)
    headway_5d = layers.Reshape(
        (lookback, num_stations, num_directions, 1),
        name="reshape_headway_5d"
    )(input_headway)
    
    # Reshape broadcasted schedule: (B, T, 66, 2) -> (B, T, 66, 2, 1)
    schedule_5d = layers.Reshape(
        (forecast, num_stations, num_directions, 1),
        name="reshape_schedule_5d"
    )(schedule_broadcast)
    
    # =========================================================================
    # Encoder: Process Historical Headways
    # =========================================================================
    # Encodes the current state of delays across the network
    
    # Encoder Layer 1
    encoder_x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        return_sequences=True,
        activation="relu",
        name="encoder_convlstm_1"
    )(headway_5d)
    
    encoder_x = layers.BatchNormalization(name="encoder_bn_1")(encoder_x)
    
    # Encoder Layer 2 - Returns states for decoder initialization
    encoder_x, state_h, state_c = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        return_sequences=True,
        return_state=True,
        activation="relu",
        name="encoder_convlstm_2"
    )(encoder_x)
    
    encoder_x = layers.BatchNormalization(name="encoder_bn_2")(encoder_x)
    
    # =========================================================================
    # Decoder: Generate Forecasts Conditioned on Future Schedule
    # =========================================================================
    # Primed with encoder states (memory of current delays)
    # Driven by broadcasted future schedule (control signal)
    
    # Decoder Layer 1 - Initialized with encoder states
    decoder_x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        return_sequences=True,
        activation="relu",
        name="decoder_convlstm_1"
    )(schedule_5d, initial_state=[state_h, state_c])
    
    decoder_x = layers.BatchNormalization(name="decoder_bn_1")(decoder_x)
    
    # Decoder Layer 2 - Additional capacity for complex propagation
    decoder_x, _, _ = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        return_sequences=True,
        return_state=True,
        activation="relu",
        name="decoder_convlstm_2"
    )(decoder_x)
    
    decoder_x = layers.BatchNormalization(name="decoder_bn_2")(decoder_x)
    
    # =========================================================================
    # Output Projection
    # =========================================================================
    # Project from filter channels back to single headway value per cell
    # Conv3D: (B, T, 66, 2, Filters) -> (B, T, 66, 2, 1)
    
    output_conv = layers.Conv3D(
        filters=1,
        kernel_size=(3, 3, 1),
        padding="same",
        activation="relu",  # Headways must be positive
        name="output_projection"
    )(decoder_x)
    
    # Reshape back to (B, Forecast, Stations, Directions)
    output = layers.Reshape(
        (forecast, num_stations, num_directions),
        name="output_reshape"
    )(output_conv)
    
    # =========================================================================
    # Build Model
    # =========================================================================
    model = keras.Model(
        inputs={"headway_input": input_headway, "schedule_input": input_schedule},
        outputs=output,
        name="ConvLSTM_HeadwayPredictor"
    )
    
    return model


# =============================================================================
# Model Summary Utility
# =============================================================================

def print_model_summary(config: Config = None):
    """Print model architecture summary for verification."""
    model = build_convlstm_model(config)
    model.summary()
    return model


if __name__ == "__main__":
    # Quick test: build and summarize model
    print("Building ConvLSTM Encoder-Decoder Model...")
    print("=" * 60)
    model = print_model_summary()
    print("\n" + "=" * 60)
    print(f"Total parameters: {model.count_params():,}")
    
    # Verify shapes match data
    print("\n" + "=" * 60)
    print("INPUT/OUTPUT SHAPE VERIFICATION")
    print("=" * 60)
    print("Model Inputs:")
    for name, spec in model.input.items():
        print(f"  {name}: {spec.shape}")
    print(f"Model Output: {model.output.shape}")
    print("\nExpected Data Shapes:")
    print("  headway_matrix_full.npy: (T, 66, 2, 1) -> (B, 30, 66, 2)")
    print("  schedule_matrix_full.npy: (T, 2, 1) -> (B, 15, 2)")

