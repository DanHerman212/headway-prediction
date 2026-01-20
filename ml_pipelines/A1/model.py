"""
Multi-Output Stacked GRU Model for A1 Track

Architecture:
- Input: (batch, 20, 8) - 20 timesteps, 8 features
- Stacked GRU layers (no dropout for initial training - intentional overfit)
- Two output heads:
  1. Classification: route_id (A, C, E) - categorical
  2. Regression: headway prediction - continuous

Handles composite headways at stations receiving multiple routes.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import config


def build_stacked_gru_model(
    lookback_window: int = None,
    num_features: int = None,
    gru_units_1: int = None,
    gru_units_2: int = None,
    dropout_rate: float = None,
    dense_units: int = None,
    num_routes: int = None
) -> keras.Model:
    """
    Build multi-output stacked GRU model.
    
    Args:
        lookback_window: Number of timesteps in input sequence
        num_features: Number of features per timestep
        gru_units_1: Units in first GRU layer
        gru_units_2: Units in second GRU layer
        dropout_rate: Dropout rate after each GRU layer
        dense_units: Units in shared dense layer
        num_routes: Number of route classes for classification
    
    Returns:
        Compiled Keras Model with two outputs
    """
    # Use config defaults if not provided
    lookback_window = lookback_window or config.LOOKBACK_WINDOW
    num_features = num_features or config.NUM_FEATURES
    gru_units_1 = gru_units_1 or config.GRU_UNITS_1
    gru_units_2 = gru_units_2 or config.GRU_UNITS_2
    dropout_rate = dropout_rate or config.DROPOUT_RATE
    dense_units = dense_units or config.DENSE_UNITS
    num_routes = num_routes or config.NUM_ROUTES
    
    print("Building Stacked GRU Model (No Dropout - Overfit Test)...")
    print(f"  Input shape: ({lookback_window}, {num_features})")
    print(f"  GRU layers: {gru_units_1} → {gru_units_2}")
    print(f"  Dense units: {dense_units}")
    print(f"  Outputs: route ({num_routes} classes), headway (1 value)")
    print(f"  Note: Dropout disabled for initial training run")
    
    # Input layer
    inputs = layers.Input(shape=(lookback_window, num_features), name='input')
    
    # First GRU layer (return sequences for stacking)
    x = layers.GRU(
        units=gru_units_1,
        return_sequences=True,
        name='gru_1'
    )(inputs)
    # No dropout for overfit test
    
    # Second GRU layer (return only final state)
    x = layers.GRU(
        units=gru_units_2,
        return_sequences=False,
        name='gru_2'
    )(x)
    # No dropout for overfit test
    
    # Shared dense layer
    x = layers.Dense(dense_units, activation='relu', name='shared_dense')(x)
    
    # Output 1: Route classification (softmax for multi-class)
    route_output = layers.Dense(
        num_routes,
        activation='softmax',
        name='route_output'
    )(x)
    
    # Output 2: Headway regression (linear activation)
    headway_output = layers.Dense(
        1,
        activation='linear',
        name='headway_output'
    )(x)
    
    # Create model with two outputs
    model = keras.Model(
        inputs=inputs,
        outputs={'route_output': route_output, 'headway_output': headway_output},
        name='stacked_gru_multioutput'
    )
    
    return model


def mae_seconds(y_true, y_pred):
    """
    Custom metric: Mean Absolute Error in seconds.
    
    Converts log-scaled headway predictions to real seconds.
    Formula: exp(pred) - 1 (inverse of log(x + 1))
    Then convert minutes to seconds.
    
    Args:
        y_true: True log-scaled headway values
        y_pred: Predicted log-scaled headway values
    
    Returns:
        MAE in seconds
    """
    # Inverse log transform to get minutes
    y_true_minutes = tf.exp(y_true) - config.LOG_OFFSET
    y_pred_minutes = tf.exp(y_pred) - config.LOG_OFFSET
    
    # Convert to seconds
    y_true_seconds = y_true_minutes * 60.0
    y_pred_seconds = y_pred_minutes * 60.0
    
    # Calculate MAE
    mae = tf.reduce_mean(tf.abs(y_true_seconds - y_pred_seconds))
    
    return mae


def compile_model(model: keras.Model, learning_rate: float = None) -> keras.Model:
    """
    Compile model with multi-output losses and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled model
    """
    learning_rate = learning_rate or config.LEARNING_RATE
    
    print("\nCompiling model...")
    print(f"  Optimizer: Adam (lr={learning_rate})")
    print(f"  Losses:")
    print(f"    route_output: {config.CLASSIFICATION_LOSS}")
    print(f"    headway_output: {config.REGRESSION_LOSS}")
    print(f"  Metrics:")
    print(f"    route_output: accuracy")
    print(f"    headway_output: mae_seconds (custom)")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'route_output': config.CLASSIFICATION_LOSS,
            'headway_output': config.REGRESSION_LOSS
        },
        loss_weights=config.LOSS_WEIGHTS,
        metrics={
            'route_output': ['accuracy'],
            'headway_output': [mae_seconds]
        }
    )
    
    return model


def get_model(compile: bool = True) -> keras.Model:
    """
    Convenience function to build and optionally compile model.
    
    Args:
        compile: Whether to compile the model
    
    Returns:
        Keras Model (compiled if compile=True)
    """
    model = build_stacked_gru_model()
    
    if compile:
        model = compile_model(model)
    
    # Print model summary
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    model.summary()
    
    return model
