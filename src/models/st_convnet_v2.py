# v2 architecture updates (FIXED VERSION)
# Key fixes from original broken V2:
# 1. Use return_sequences=False in encoder to create bottleneck (like V1)
# 2. Repeat state instead of slicing (proper seq2seq pattern)
# 3. Single decoder ConvLSTM (reduce depth, improve gradient flow)
# 4. Sigmoid output for [0,1] normalized data
# 5. Removed redundant skip connection

import tensorflow as tf
from tensorflow.keras import layers, models, Input, backend as K
from tensorflow.keras.layers import(
    ConvLSTM2D, 
    BatchNormalization,
    Conv3D,
    Concatenate,
    Lambda
)

class HeadwayConvLSTM:
    def __init__(self, n_stations, lookback=30, forecast=15, output_activation='linear'):
        """
        Args:
            n_stations: Number of stations in the network
            lookback: Historical timesteps (default 30)
            forecast: Future timesteps to predict (default 15)
            output_activation: 'sigmoid' for [0,1] normalized data, 'linear' for RobustScaler
        """
        self.n_stations = n_stations
        self.lookback = lookback
        self.forecast = forecast
        self.output_activation = output_activation

    def build_model(self):
        """
        Builds the optimized spatio-temporal convolutional LSTM model (V2 - Fixed).

        Architecture: Encoder-Bottleneck-Decoder with schedule fusion
        - Encoder: 2 ConvLSTM layers, second compresses to single state
        - Bridge: Repeat state for forecast steps (NOT temporal slicing)
        - Schedule: Asymmetric Conv3D kernels for space/time processing
        - Decoder: Single ConvLSTM layer (prevents over-depth)
        - Output: Sigmoid activation for [0,1] normalized headways
        """
        # === INPUTS ===
        # Headway history: (batch, lookback, stations, directions, 1)
        input_headway = Input(shape=(self.lookback, self.n_stations, 2, 1), name="headway_input")
        
        # Future schedule: (batch, forecast, directions, 1)
        input_schedule = Input(shape=(self.forecast, 2, 1), name="schedule_input")

        # === ENCODER (CuDNN Accelerated) ===
        # Layer 1: Extract low-level spatiotemporal features
        x = ConvLSTM2D(
            filters=32, 
            kernel_size=(3, 3), 
            padding="same",
            return_sequences=True,  # Keep sequence for layer 2
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0
        )(input_headway)
        x = BatchNormalization()(x)

        # Layer 2: Compress to single context state (CRITICAL FIX)
        # return_sequences=False creates the bottleneck that enables learning
        state = ConvLSTM2D(
            filters=64, 
            kernel_size=(3, 3), 
            padding="same",
            return_sequences=False,  # ← KEY: Compress 30 steps to 1 state
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0
        )(x)
        state = BatchNormalization()(state)
        # state shape: (batch, stations, 2, 64)

        # === BRIDGE (Repeat State) ===
        # Expand context state into forecast timesteps
        def repeat_state(tensor):
            # tensor: (batch, stations, dirs, filters)
            expanded = tf.expand_dims(tensor, axis=1)  # (batch, 1, stations, dirs, filters)
            return tf.tile(expanded, [1, self.forecast, 1, 1, 1])  # (batch, forecast, ...)
        
        state_repeated = Lambda(
            repeat_state,
            output_shape=(self.forecast, self.n_stations, 2, 64),
            name="repeat_state"
        )(state)
        # state_repeated shape: (batch, 15, stations, 2, 64)

        # === SCHEDULE PROCESSING (Asymmetric Kernels) ===
        # Broadcast schedule to match spatial dimensions
        # (batch, forecast, 2, 1) → (batch, forecast, stations, 2, 1)
        sch_broadcast = Lambda(
            lambda x: K.tile(K.expand_dims(x, axis=2), [1, 1, self.n_stations, 1, 1]),
            output_shape=(self.forecast, self.n_stations, 2, 1),
            name="broadcast_schedule"
        )(input_schedule)
        
        # Spatial convolution: learn cross-station patterns
        sch_spatial = Conv3D(
            filters=8, 
            kernel_size=(1, 3, 3),  # No temporal, 3x3 spatial
            padding="same",
            activation="relu", 
            name="sched_spatial_conv"
        )(sch_broadcast)
        
        # Temporal convolution: learn time dynamics
        sch_features = Conv3D(
            filters=16, 
            kernel_size=(3, 1, 1),  # 3 temporal, no spatial
            padding="same",
            activation="relu", 
            name="sched_temporal_conv"
        )(sch_spatial)
        # sch_features shape: (batch, 15, stations, 2, 16)

        # === FUSION ===
        # Combine encoded state with schedule features
        # (batch, 15, stations, 2, 64) + (batch, 15, stations, 2, 16) → 80 channels
        decoder_input = Concatenate(axis=-1, name="fusion_concat")([state_repeated, sch_features])

        # === DECODER (Single Layer - Prevents Over-Depth) ===
        decoded = ConvLSTM2D(
            filters=32, 
            kernel_size=(3, 3), 
            padding="same",
            return_sequences=True,  # Output all 15 timesteps
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0
        )(decoder_input)
        decoded = BatchNormalization()(decoded)
        # decoded shape: (batch, 15, stations, 2, 32)

        # === OUTPUT ===
        # Project to single channel
        # Use 'sigmoid' for [0,1] MinMax normalized data
        # Use 'linear' for RobustScaler or StandardScaler (unbounded range)
        output = Conv3D(
            filters=1, 
            kernel_size=(1, 1, 1), 
            activation=self.output_activation,
            padding="same", 
            name="headway_output"
        )(decoded)
        # output shape: (batch, 15, stations, 2, 1)
        
        model = models.Model(
            inputs=[input_headway, input_schedule], 
            outputs=output, 
            name="HeadwayConvLSTM_V2"
        )

        return model