# v2 architecture updates
# 1 encoder - sequential convlstm structure
# 2 batch norm - correctly placed between convolution and activation
# 3 schedule processing - assymetric conv3d kernels, spatial (1, 3, 3) the temporal (3, 1, 1)
# 4 skip connections - schedule features inject at fusion and output
# 5 output linear activation avoid saturation
import tensorflow as tf
from tensorflow.keras import layers, models, Input, backend as K
from tensorflow.keras.layers import(
    ConvLSTM2D, 
    BatchNormalization,
    Conv3D,
    Concatenate,
    TimeDistributed,
    Activation,
    Lambda
)

class HeadwayConvLSTM:
    def __init__(self, n_stations, lookback=30, forecast=15):
        self.n_stations = n_stations
        self.lookback = lookback
        self.forecast = forecast

    def build_model(self):
        """
        Builds the optimized spatio-temporal convulational lstm model (V2).

        Key improvements:
        - Asymmetric kernels for scheudle processing (spearating space/time)
        - BatchNormalization applied before activation
        - Skip connections for residual learning
        - Linear output activation
        """
        # inputs
        # shape (batch, lookback, stations, direcctions, 1)
        input_headway = Input(shape=(self.lookback, self.n_stations, 2, 1), name="headway_input")

        # shape: (batch, forecasts, distrctions, 1) -> will be broadcasted
        input_schedule = Input(shape=(self.forecast, 2, 1), name="schedule_input")

        # encoder ( fully sequential)
        # layer 1: low level features
        # note: activation=None allows us to place BN before the activation fucntion
        x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same',
                       return_sequences=True, activation=None)(input_headway)
        x = TimeDistributed(BatchNormalization())(x)
        x = Activation('tanh')(x)

        # layer 2: high level features and time compression
        # return_sequences=True to preserve temporal dynamics across all 30 steps
        x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same",
                       return_sequences=True, activation=None)(x)
        x = TimeDistributed(BatchNormalization())(x)
        encoded_sequence = Activation('tanh')(x) # shape (batch, 30, stations, 2, 64)

        # bridge (Temporal Slicing)
        # instead of repeating a static state, we slice the last 'forecast' steps
        # this feeds the decoder the actual motion/trends of the most recent 15 minutes
        def slice_last_steps(x):
            # x_shape: (batch, lookback, H, W, C) -> take last 'forecast' steps
            return x[:, -self.forecast:, :, :, :]
        
        # output shape: (batch, 15, stations, 2 64)
        bridge_output = Lambda(slice_last_steps,
                               output_shape=(self.forecast, self.n_stations, 2, 64),
                               name="slice_temporal_state")(encoded_sequence)
        
        # schedule processing (assymetic kernels)
        # 1. broadcast scheudle to match spatial dimensions
        # (batch, Time, 2, 1) -> (Batch, Time Stations, 2, 1)
        sch_broadcast = Lambda(lambda x: K.tile(K.expand_dims(x, axis=2), [1, 1, self.n_stations, 1, 1]),
                               output_shape=(self.forecast, self.n_stations, 2, 1),
                               name="broadcast_schedule")(input_schedule)
        
        # 2 factored convolution (the pro experiment)
        # branch A: spatial correlations (1, 3, 3)
        # looks at neighboring stations at the same time step
        sch_spatial = Conv3D(filters=8, kernel_size=(1, 3, 3), padding="same",
                             activation="relu", name="sched_spatial_conv")(sch_broadcast)
        
        # branch B: Temporal dynamics (3, 1, 1) on top of spatial features
        # looks at history of specific station/direction
        sch_features = Conv3D(filters=16, kernel_size=(3, 1, 1), padding="same",
                              activation="relu", name="sched_temporal_conv")(sch_spatial)
        
        # fusion
        # combine the recent historical motion (bridge) with future schedule
        decoder_input = Concatenate(axis=-1, name="fusion_concat")([bridge_output, sch_features])

        # decoder
        # layer 1 refine predictions based on combined state
        d = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same",
                       return_sequences=True, activation=None)(decoder_input)
        d = TimeDistributed(BatchNormalization())(d)
        d = Activation("tanh")(d)

        # layer 2 final feature extraction
        d = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same",
                       return_sequences=True, activation=None)(d)
        d = TimeDistributed(BatchNormalization())(d)
        decoded = Activation("tanh")(d)

        # skip connection and output
        # inject scheudle features again so the model directly sees the baseline it should deviate from
        final_input_stack = Concatenate(axis=-1, name="skip_connection")([decoded, sch_features])

        # final projection to signal (regression)
        # linear activation allows for unbound values
        output = Conv3D(filters=1, kernel_size=(1, 1, 1), activation="linear",
                        padding="same", name="main_output")(final_input_stack)
        
        model = models.Model(inputs=[input_headway, input_schedule], outputs=output, name="HeadwayConvLSTM_V2")

        return model