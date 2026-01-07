# Paper-Faithful Baseline ConvLSTM
# Based on: Usama & Koutsopoulos (2025)
# "Real Time Headway Predictions in Urban Rail Systems"
# arXiv:2510.03121
#
# This is the GROUND TRUTH baseline. All experiments compare against this.
#
# Paper hyperparameters (Table 1):
# - Lookback: 30 min, Forecast: 15 min
# - Filters: 32, Kernel: 3x3
# - Activation: tanh (CuDNN), sigmoid output
# - MinMax normalization [0,1]

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from src.config import Config


class HeadwayConvLSTM:
    """
    Paper-faithful ConvLSTM for headway prediction.
    
    Architecture (from Figure 2):
        Encoder:  ConvLSTM(32) -> BN -> ConvLSTM(32) -> BN
        Bridge:   Repeat state for forecast horizon
        Fusion:   Broadcast schedule + concatenate
        Decoder:  ConvLSTM(32) -> BN
        Output:   Conv3D(1) with sigmoid
    
    This class follows the established pattern:
    - Takes Config for hyperparameters
    - Only defines build_model() - no compilation
    - Trainer handles compilation and training
    """
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object with hyperparameters
        """
        self.config = config
    
    def build_model(self) -> Model:
        """
        Build the paper-faithful ConvLSTM model.
        
        Input shapes (from data pipeline):
            headway_input:  (batch, lookback, stations, 2, 1)
            schedule_input: (batch, forecast, 2, 1)
        
        Output shape:
            (batch, forecast, stations, 2, 1)
        
        Returns:
            Keras Model (uncompiled)
        """
        lookback = self.config.LOOKBACK_MINS
        forecast = self.config.FORECAST_MINS
        n_stations = self.config.NUM_STATIONS
        filters = self.config.FILTERS
        kernel_size = self.config.KERNEL_SIZE
        
        # ====================================================================
        # INPUTS
        # ====================================================================
        input_headway = Input(
            shape=(lookback, n_stations, 2, 1),
            name="headway_input"
        )
        
        input_schedule = Input(
            shape=(forecast, 2, 1),
            name="schedule_input"
        )
        
        # ====================================================================
        # ENCODER
        # ====================================================================
        # Layer 1: Process full sequence
        # CuDNN requirements: activation='tanh', recurrent_activation='sigmoid'
        x = layers.ConvLSTM2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            name='encoder_1'
        )(input_headway)
        x = layers.BatchNormalization(name='encoder_bn_1')(x)
        
        # Layer 2: Compress to single state
        state = layers.ConvLSTM2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            return_sequences=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            name='encoder_2'
        )(x)
        state = layers.BatchNormalization(name='encoder_bn_2')(state)
        # state shape: (batch, stations, 2, filters)
        
        # ====================================================================
        # BRIDGE
        # ====================================================================
        def repeat_state(tensor):
            expanded = tf.expand_dims(tensor, axis=1)
            return tf.tile(expanded, [1, forecast, 1, 1, 1])
        
        state_repeated = layers.Lambda(
            repeat_state,
            output_shape=(forecast, n_stations, 2, filters),
            name='repeat_state'
        )(state)
        
        # ====================================================================
        # SCHEDULE FUSION
        # ====================================================================
        def broadcast_schedule(args):
            schedule, ref = args
            expanded = tf.expand_dims(schedule, axis=2)
            n_stat = tf.shape(ref)[2]
            return tf.tile(expanded, [1, 1, n_stat, 1, 1])
        
        schedule_broadcast = layers.Lambda(
            broadcast_schedule,
            output_shape=(forecast, n_stations, 2, 1),
            name='broadcast_schedule'
        )([input_schedule, state_repeated])
        
        decoder_input = layers.Concatenate(
            axis=-1, 
            name='fusion'
        )([state_repeated, schedule_broadcast])
        
        # ====================================================================
        # DECODER
        # ====================================================================
        decoded = layers.ConvLSTM2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            name='decoder'
        )(decoder_input)
        decoded = layers.BatchNormalization(name='decoder_bn')(decoded)
        
        # ====================================================================
        # OUTPUT
        # ====================================================================
        output = layers.Conv3D(
            filters=1,
            kernel_size=(1, 1, 1),
            activation='sigmoid',
            padding='same',
            name='output'
        )(decoded)
        
        # ====================================================================
        # MODEL
        # ====================================================================
        model = Model(
            inputs=[input_headway, input_schedule],
            outputs=output,
            name='BaselineConvLSTM'
        )
        
        return model
