"""
Paper-Faithful Baseline ConvLSTM Model

Minimal implementation matching Usama & Koutsopoulos (2025):
"Real Time Headway Predictions in Urban Rail Systems"
arXiv:2510.03121

Key design principles:
1. EXACTLY match paper hyperparameters (32 filters, 3x3 kernel)
2. NO experimental modifications (no extra dropout, no fancy optimizers)
3. Clean, readable code for debugging and iteration
4. Full documentation of architecture choices

This is the GROUND TRUTH baseline. All future experiments should
compare against this model's performance.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input


class BaselineConvLSTM:
    """
    Paper-faithful ConvLSTM for headway prediction.
    
    Architecture (from Table 1 and Figure 2):
        Encoder:  ConvLSTM(32) -> BN -> ConvLSTM(32) -> BN
        Bridge:   Repeat state for forecast horizon
        Fusion:   Broadcast schedule + concatenate
        Decoder:  ConvLSTM(32) -> BN
        Output:   Conv3D(1) with sigmoid
    
    Hyperparameters (Table 1):
        - Lookback: 30 min
        - Forecast: 15 min
        - Filters: 32
        - Kernel: 3x3
        - Activation: tanh (for CuDNN), ReLU post-BN
        - Output: sigmoid (MinMax [0,1] data)
        - Batch size: 32
        - Epochs: 100
        - Early stopping: 50 epochs patience
    """
    
    # Paper hyperparameters (Table 1)
    PAPER_CONFIG = {
        'lookback': 30,
        'forecast': 15,
        'filters': 32,
        'kernel_size': (3, 3),
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 50,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss': 'mse',
    }
    
    def __init__(
        self, 
        n_stations: int,
        lookback: int = 30,
        forecast: int = 15,
        filters: int = 32,
        kernel_size: tuple = (3, 3),
    ):
        """
        Initialize baseline model.
        
        Args:
            n_stations: Number of stations (spatial dimension)
            lookback: Historical timesteps (default: 30 per paper)
            forecast: Future timesteps to predict (default: 15 per paper)
            filters: ConvLSTM filters (default: 32 per paper)
            kernel_size: Convolution kernel size (default: 3x3 per paper)
        """
        self.n_stations = n_stations
        self.lookback = lookback
        self.forecast = forecast
        self.filters = filters
        self.kernel_size = kernel_size
    
    def build_model(self) -> Model:
        """
        Build the paper-faithful ConvLSTM model.
        
        Architecture follows Figure 2 exactly:
        1. Input historical headways: (batch, lookback, stations, directions, 1)
        2. Input future schedule: (batch, forecast, directions, 1)
        3. Encoder: Extract spatiotemporal features
        4. Bridge: Repeat encoded state
        5. Fusion: Incorporate schedule information
        6. Decoder: Generate predictions
        7. Output: Sigmoid-activated headway predictions
        
        Returns:
            Compiled Keras Model
        """
        # ====================================================================
        # INPUTS
        # ====================================================================
        # Historical headways: X ∈ R^(L×Nd×Ndir×1)
        input_headway = Input(
            shape=(self.lookback, self.n_stations, 2, 1),
            name="headway_input"
        )
        
        # Future terminal schedule: T ∈ R^(F×Ndir×1)
        input_schedule = Input(
            shape=(self.forecast, 2, 1),
            name="schedule_input"
        )
        
        # ====================================================================
        # ENCODER
        # ====================================================================
        # Layer 1: Process full sequence
        # CuDNN requirements: activation='tanh', recurrent_activation='sigmoid'
        x = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,  # CuDNN requires 0
            name='encoder_1'
        )(input_headway)
        x = layers.BatchNormalization(name='encoder_bn_1')(x)
        
        # Layer 2: Compress to single state
        state = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            return_sequences=False,  # Output single state
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
        # Repeat encoded state for each forecast timestep
        def repeat_state(tensor):
            # Add time dimension and tile
            expanded = tf.expand_dims(tensor, axis=1)
            return tf.tile(expanded, [1, self.forecast, 1, 1, 1])
        
        state_repeated = layers.Lambda(
            repeat_state,
            output_shape=(self.forecast, self.n_stations, 2, self.filters),
            name='repeat_state'
        )(state)
        # shape: (batch, forecast, stations, 2, filters)
        
        # ====================================================================
        # SCHEDULE FUSION
        # ====================================================================
        # Broadcast schedule to match spatial dimensions
        def broadcast_schedule(args):
            schedule, ref = args
            # schedule: (batch, forecast, 2, 1)
            # Expand: (batch, forecast, 1, 2, 1)
            expanded = tf.expand_dims(schedule, axis=2)
            # Tile across stations: (batch, forecast, stations, 2, 1)
            n_stations = tf.shape(ref)[2]
            return tf.tile(expanded, [1, 1, n_stations, 1, 1])
        
        schedule_broadcast = layers.Lambda(
            broadcast_schedule,
            output_shape=(self.forecast, self.n_stations, 2, 1),
            name='broadcast_schedule'
        )([input_schedule, state_repeated])
        
        # Concatenate along channel axis
        # (batch, forecast, stations, 2, filters) + (batch, forecast, stations, 2, 1)
        # = (batch, forecast, stations, 2, filters+1)
        decoder_input = layers.Concatenate(
            axis=-1, 
            name='fusion'
        )([state_repeated, schedule_broadcast])
        
        # ====================================================================
        # DECODER
        # ====================================================================
        decoded = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            name='decoder'
        )(decoder_input)
        decoded = layers.BatchNormalization(name='decoder_bn')(decoded)
        # shape: (batch, forecast, stations, 2, filters)
        
        # ====================================================================
        # OUTPUT
        # ====================================================================
        # Project to single channel with sigmoid (data normalized to [0,1])
        output = layers.Conv3D(
            filters=1,
            kernel_size=(1, 1, 1),
            activation='sigmoid',
            padding='same',
            name='output'
        )(decoded)
        # output shape: (batch, forecast, stations, 2, 1)
        
        # ====================================================================
        # MODEL
        # ====================================================================
        model = Model(
            inputs=[input_headway, input_schedule],
            outputs=output,
            name='BaselineConvLSTM'
        )
        
        return model
    
    def get_compile_args(self, learning_rate: float = None) -> dict:
        """
        Get paper-faithful compilation arguments.
        
        Args:
            learning_rate: Override learning rate (default: 0.001 per paper)
            
        Returns:
            Dict with optimizer, loss, metrics
        """
        lr = learning_rate or self.PAPER_CONFIG['learning_rate']
        
        return {
            'optimizer': tf.keras.optimizers.Adam(learning_rate=lr),
            'loss': 'mse',
        }
    
    def get_training_args(self) -> dict:
        """
        Get paper-faithful training arguments.
        
        Returns:
            Dict with epochs, batch_size, early_stopping settings
        """
        return {
            'epochs': self.PAPER_CONFIG['epochs'],
            'batch_size': self.PAPER_CONFIG['batch_size'],
            'early_stopping_patience': self.PAPER_CONFIG['early_stopping_patience'],
        }
    
    @classmethod
    def from_paper_config(cls, n_stations: int) -> 'BaselineConvLSTM':
        """
        Create model with exact paper hyperparameters.
        
        Args:
            n_stations: Number of stations
            
        Returns:
            BaselineConvLSTM instance
        """
        return cls(
            n_stations=n_stations,
            lookback=cls.PAPER_CONFIG['lookback'],
            forecast=cls.PAPER_CONFIG['forecast'],
            filters=cls.PAPER_CONFIG['filters'],
            kernel_size=cls.PAPER_CONFIG['kernel_size'],
        )


def count_params(model: Model) -> dict:
    """
    Count model parameters.
    
    Args:
        model: Keras model
        
    Returns:
        Dict with total, trainable, non_trainable counts
    """
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    
    return {
        'total': trainable + non_trainable,
        'trainable': trainable,
        'non_trainable': non_trainable,
    }


# Quick test when run directly
if __name__ == "__main__":
    print("Building BaselineConvLSTM with paper configuration...")
    
    builder = BaselineConvLSTM.from_paper_config(n_stations=66)
    model = builder.build_model()
    
    compile_args = builder.get_compile_args()
    model.compile(**compile_args)
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nParameter Counts:")
    params = count_params(model)
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")
    
    print("\nPaper Configuration:")
    for key, value in builder.PAPER_CONFIG.items():
        print(f"  {key}: {value}")
