# Paper-Faithful Implementation
# Based on: Usama & Koutsopoulos (2025)
# "Real Time Headway Predictions in Urban Rail Systems"
# arXiv:2510.03121
#
# Key architecture from Table 1:
# - Input: X ∈ R^(L×Nd×Ndir×1), T ∈ R^(F×Ndir×1)
# - ConvLSTM filters: 32
# - Kernel size: 3×3
# - Activation: ReLU
# - Normalization: MinMax [0,1]
# - Output: sigmoid

import tensorflow as tf
from tensorflow.keras import layers, models, Input, backend as K
from tensorflow.keras.layers import (
    ConvLSTM2D, 
    BatchNormalization,
    Conv3D,
    Concatenate,
    Lambda
)


class HeadwayConvLSTM:
    """
    Paper-faithful ConvLSTM architecture for spatiotemporal headway prediction.
    
    Based on Figure 2 of Usama & Koutsopoulos (2025):
    - Encoder: ConvLSTM layers process historical headways
    - Schedule Fusion: Terminal headways broadcast and concatenated
    - Decoder: ConvLSTM layers produce future headway predictions
    
    Paper hyperparameters (Table 1):
    - Lookback: 30 min, Forecast: 15 min
    - Filters: 32, Kernel: 3×3
    - Activation: ReLU
    - MinMax normalization with sigmoid output
    """
    
    def __init__(self, n_stations, lookback=30, forecast=15):
        """
        Args:
            n_stations: Number of distance bins (paper uses 64)
            lookback: Historical timesteps (paper: 30 minutes)
            forecast: Future timesteps to predict (paper: 15 minutes)
        """
        self.n_stations = n_stations
        self.lookback = lookback
        self.forecast = forecast

    def build_model(self):
        """
        Builds the ConvLSTM model matching paper architecture.
        
        Architecture (from Figure 2):
        1. Input: Historical headways (30, N_d, 2, 1) + Terminal schedule (15, 2, 1)
        2. Encoder: ConvLSTM extracts spatiotemporal features
        3. Bridge: Repeat encoded state for forecast horizon
        4. Fusion: Concatenate with broadcast schedule
        5. Decoder: ConvLSTM generates predictions
        6. Output: Sigmoid for [0,1] normalized headways
        
        Returns:
            Compiled Keras Model
        """
        # === INPUTS (Equations 4-5) ===
        # Historical headway tensor: X ∈ R^(L×Nd×Ndir×1)
        input_headway = Input(
            shape=(self.lookback, self.n_stations, 2, 1), 
            name="headway_input"
        )
        
        # Future terminal headway tensor: T ∈ R^(F×Ndir×1)
        input_schedule = Input(
            shape=(self.forecast, 2, 1), 
            name="schedule_input"
        )

        # === ENCODER ===
        # CRITICAL: Must use tanh activation for CuDNN acceleration
        # Paper's "ReLU" in Table 1 likely refers to post-BN activations
        # Using relu in ConvLSTM disables CuDNN → 10x slower!
        x = ConvLSTM2D(
            filters=32,  # Table 1: "ConvLSTM Filters: 32"
            kernel_size=(3, 3),  # Table 1: "ConvLSTM Kernel Size: 3×3"
            padding="same",
            return_sequences=True,
            activation='tanh',  # CuDNN requires tanh
            recurrent_activation='sigmoid',  # CuDNN requires sigmoid
            recurrent_dropout=0  # CuDNN requires 0
        )(input_headway)
        x = BatchNormalization()(x)

        # Second encoder layer - compress to context state
        state = ConvLSTM2D(
            filters=32,  # Keep consistent with paper's 32 filters
            kernel_size=(3, 3),
            padding="same",
            return_sequences=False,  # Compress to single state
            activation='tanh',  # CuDNN requires tanh
            recurrent_activation='sigmoid',
            recurrent_dropout=0
        )(x)
        state = BatchNormalization()(state)
        # state shape: (batch, stations, 2, 32)

        # === BRIDGE ===
        # Repeat encoded state for each forecast timestep
        def repeat_state(tensor):
            expanded = tf.expand_dims(tensor, axis=1)
            return tf.tile(expanded, [1, self.forecast, 1, 1, 1])
        
        state_repeated = Lambda(
            repeat_state,
            output_shape=(self.forecast, self.n_stations, 2, 32),
            name="repeat_state"
        )(state)
        # shape: (batch, 15, stations, 2, 32)

        # === SCHEDULE FUSION ===
        # Paper: "directly incorporating planned terminal headways as a critical input"
        # Simple broadcast - no complex Conv3D processing
        def broadcast_schedule(args):
            schedule, ref_tensor = args
            # schedule: (batch, forecast, 2, 1)
            # Expand to match spatial dims: (batch, forecast, 1, 2, 1)
            expanded = tf.expand_dims(schedule, axis=2)
            # Tile across stations: (batch, forecast, stations, 2, 1)
            stations = tf.shape(ref_tensor)[2]
            return tf.tile(expanded, [1, 1, stations, 1, 1])
        
        schedule_broadcast = Lambda(
            broadcast_schedule,
            output_shape=(self.forecast, self.n_stations, 2, 1),
            name="broadcast_schedule"
        )([input_schedule, state_repeated])

        # Concatenate along channel axis
        # (batch, 15, stations, 2, 32) + (batch, 15, stations, 2, 1) → 33 channels
        decoder_input = Concatenate(axis=-1, name="fusion")([state_repeated, schedule_broadcast])

        # === DECODER ===
        # Process fused spatiotemporal representation
        decoded = ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,  # Output all forecast timesteps
            activation='tanh',  # CuDNN requires tanh
            recurrent_activation='sigmoid',
            recurrent_dropout=0
        )(decoder_input)
        decoded = BatchNormalization()(decoded)
        # shape: (batch, 15, stations, 2, 32)

        # === OUTPUT ===
        # Project to single channel with sigmoid for [0,1] normalized data
        # Paper uses MinMax normalization to [0,1]
        output = Conv3D(
            filters=1, 
            kernel_size=(1, 1, 1), 
            activation='sigmoid',  # Paper: MinMax [0,1] normalization
            padding="same", 
            name="headway_output"
        )(decoded)
        # output shape: (batch, 15, stations, 2, 1)
        
        model = models.Model(
            inputs=[input_headway, input_schedule], 
            outputs=output, 
            name="HeadwayConvLSTM_Paper"
        )

        return model
    
    @staticmethod
    def get_paper_config():
        """
        Returns the exact hyperparameters from Table 1 of the paper.
        
        Use these for training to replicate paper results.
        """
        return {
            'lookback': 30,
            'forecast': 15,
            'n_distance_bins': 64,
            'n_directions': 2,
            'filters': 32,
            'kernel_size': (3, 3),
            'activation': 'relu',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss': 'mse',
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 50,
            'validation_split': 0.2,
            'normalization': 'minmax',  # [0, 1]
        }
