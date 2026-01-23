"""
Custom Metrics for ML Pipeline

Provides domain-specific metrics for model evaluation.
All metrics are compatible with TensorFlow/Keras and can be used
during training and evaluation.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="MLPipelineMetrics")
def rmse_seconds(y_true, y_pred):
    """
    Root Mean Squared Error converted to seconds.
    
    Assumes data is normalized to [0, 1] range where 1.0 represents 30 minutes.
    This provides interpretable error metrics in real-world units.
    
    Args:
        y_true: True values (normalized)
        y_pred: Predicted values (normalized)
        
    Returns:
        RMSE in seconds
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rmse = tf.sqrt(mse)
    
    # Conversion: 1.0 (normalized) = 30 mins = 1800 seconds
    return rmse * 1800.0


@tf.keras.utils.register_keras_serializable(package="MLPipelineMetrics")
def mae_seconds(y_true, y_pred):
    """
    Mean Absolute Error converted to seconds.
    
    Assumes data is normalized to [0, 1] range where 1.0 represents 30 minutes.
    
    Args:
        y_true: True values (normalized)
        y_pred: Predicted values (normalized)
        
    Returns:
        MAE in seconds
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Conversion: 1.0 (normalized) = 30 mins = 1800 seconds
    return mae * 1800.0


@tf.keras.utils.register_keras_serializable(package="MLPipelineMetrics")
def r_squared(y_true, y_pred):
    """
    R² (coefficient of determination).
    
    R² = 1 - (SS_res / SS_tot)
    
    - R² = 1.0: Perfect prediction
    - R² = 0.0: Model is as good as predicting the mean
    - R² < 0.0: Model is worse than predicting the mean
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R² score
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Residual sum of squares
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    
    # Total sum of squares
    y_mean = tf.reduce_mean(y_true)
    ss_tot = tf.reduce_sum(tf.square(y_true - y_mean))
    
    # R² with epsilon to avoid division by zero
    return 1.0 - (ss_res / (ss_tot + tf.keras.backend.epsilon()))


@tf.keras.utils.register_keras_serializable(package="MLPipelineMetrics")
def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.
    
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE in percentage
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Avoid division by zero
    epsilon = tf.keras.backend.epsilon()
    diff = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), epsilon))
    
    return 100.0 * tf.reduce_mean(diff)


# Metric registry for easy access
METRICS_REGISTRY = {
    "rmse_seconds": rmse_seconds,
    "mae_seconds": mae_seconds,
    "r_squared": r_squared,
    "mape": mape,
}


def get_metric(metric_name: str):
    """
    Get metric function by name.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Metric function
        
    Raises:
        ValueError if metric not found
    """
    if metric_name in METRICS_REGISTRY:
        return METRICS_REGISTRY[metric_name]
    else:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(METRICS_REGISTRY.keys())}")


def get_metrics(metric_names: list) -> list:
    """
    Get multiple metric functions by name.
    
    Args:
        metric_names: List of metric names
        
    Returns:
        List of metric functions
    """
    return [get_metric(name) for name in metric_names]
