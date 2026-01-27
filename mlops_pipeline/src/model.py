import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from omegaconf import DictConfig

def build_model(lookback_steps: int, n_features: int, num_routes: int, cfg: DictConfig) -> keras.Model:
    """Builds the Stacked GRU Model with Dual Outputs."""
    
    inputs = layers.Input(shape=(lookback_steps, n_features), name='input_sequence')
    
    x = inputs
    
    # Tunable GRU Layers
    gru_units = cfg.model.gru_units
    dropout_rate = cfg.model.dropout_rate
    
    for i, units in enumerate(gru_units):
        return_sequences = (i < len(gru_units) - 1)
        x = layers.GRU(
            units,
            return_sequences=return_sequences,
            name=f'gru_{i}'
        )(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Dual Output Heads
    
    # 1. Regression Head (Headway)
    headway_out = layers.Dense(1, name='headway')(x)
    
    # 2. Classification Head (Route)
    route_out = layers.Dense(num_routes, activation='softmax', name='route')(x)
    
    model = keras.Model(inputs=inputs, outputs=[headway_out, route_out], name=cfg.model.name)
    return model
