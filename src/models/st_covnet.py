# SpatioTemporal Covnet Architecture
# Encoder: Reads the 30-min history and compresses it into context state preserving spacial structure
# Repeater: Duplicates this state 15 times (for the 15 future minutes)
# Fusion: Injexts the Future Schedule into these 15 steps so the model knows where trains are expected
# Decoder: Refines this combined state into the final prediction

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.config import Config

class HeadwayConvLSTM:
    def __init__(self, config: Config):
        self.config = config

    def build_model(self):
        """
        Constructs the dual input ST-ConvNet
        Input 1: Headway History (Batch, 30, Stations, 2, 1)
        Input 2: Future Schedule (Batch 15, 2, 1)
        Output: Future Headway (Batch, 15, Stations, 2, 1)
        """

        # inputs
        # we use None for stations to allow flexiblity (156 stations)
        # shape: (Time, Stations, Directions, Channels)
        input_headway = layers.Input(
            shape=(self.config.LOOKBACK_MINS, None, 2, 1),
            name="headway_input"
        )

        # shape (Time, Directions, Channels) - No station dim in schedule
        input_schedule = layers.Input(
            shape=(self.config.FORECAST_MINS, 2, 1),
            name="schedule_input"
        )

        # encoder (past)
        # process the 30-min history
        # output (batch, 30, stations, 2, 32)
        x = layers.ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='tanh'
        )(input_headway)

        x = layers.BatchNormalization()(x)

        # compress time to a single spatial state
        # output (Batch, STations, 2, 64)
        state = layers.ConvLSTM2D(
            filters=self.config.FILTERS,
            kernel_size=self.config.KERNEL_SIZE,
            padding='same',
            return_sequences=False, # last time step only
            activation='tanh'
        )(x)

        # bridge (repeat state)
        # we need to project the static state into the future 15 steps
        forecast_steps = self.config.FORECAST_MINS

        # custom lambda to repeat the 4D tensor along a new time axis
        def repeat_spatial_state(tensor, steps):
            # tensor: (Batch, Stations, Dirs, Filters)
            # add time dim: (Batch, 1, Stations, Dirs, Filters)
            expanded = tf.expand_dims(tensor, axis=1)
            # tile: (Batch, Steps, Stations, Dirs, Filters)
            return tf.tile(expanded, [1, steps, 1, 1, 1])
        
        x_repeated = layers.Lambda(
            repeat_spatial_state, 
            arguments={'steps': forecast_steps},
            name="repeat_state"
        )([state])

        # fusion (add schedule)
        # we need to broadcast the (batch 15, 2, 1) schedule to match (Batch, 15, 2, 1)

        def broadcast_schedule(args):
            sched, ref_tensor = args
            # sched: (Batch, Time, Directions, 1)
            # ref: (Batch, Time, Stations, Directions, Filters)
            stations = tf.shape(ref_tensor)[2]
            
            # expand dims to insert stations: (Batch, Time, 1, Dirs, 1)
            sched_exp = tf.expand_dims(sched, axis=2)
            # tile along station dim
            return tf.tile(sched_exp, [1, 1, stations, 1, 1])
        
        schedule_broadcasted = layers.Lambda(
            broadcast_schedule,
            name="broadcast_schedule"
        )([input_schedule, x_repeated])

        # concatenate along channel axis
        # (Batch, 15, Stations, 2, 64) + (Batch, 15, Stations, 2, 1) -> 65 Channels
        x_fused = layers.Concatenate(axis=-1)([x_repeated, schedule_broadcasted])

        # decoder (future)
        # process the fused future sequence
        x = layers.ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='tanh'
        )(x_fused)

        # final projection to 1 channel (headway)
        # we use conv3D to map features to output value (0 - 1 range)
        outputs = layers.Conv3D(
            filters=1,
            kernel_size=(1, 1, 1),
            activation='sigmoid', # Sigmoid because data is normalized
            padding='same',
            name="headway_output"
        )(x)

        model = keras.Model(inputs=[input_headway, input_schedule], outputs=outputs)

        return model