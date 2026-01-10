import tensorflow as tf

# Use tf.keras.utils for compatibility with TF 2.x versions
@tf.keras.utils.register_keras_serializable(package="HeadwayMetrics")
def rmse_seconds(y_true, y_pred):
    """
    Computes Root Mean Squared Error (RMSE) converted to seconds.
    Assumes data is normalized such that 1.0 = 30 minutes.
    """
    # Ensure float32 for stability, especially if mixed_precision is on
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rmse = tf.sqrt(mse)
    
    # Conversion: 1.0 (norm) = 30 mins = 1800 seconds
    return rmse * 1800.0

@tf.keras.utils.register_keras_serializable(package="HeadwayMetrics")
def mae_seconds(y_true, y_pred):
    """
    Computes Mean Absolute Error (MAE) converted to seconds.
    Assumes data is normalized such that 1.0 = 30 minutes.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Conversion: 1.0 (norm) = 30 mins = 1800 seconds
    return mae * 1800.0


@tf.keras.utils.register_keras_serializable(package="HeadwayMetrics")
def r_squared(y_true, y_pred):
    """
    Computes R² (coefficient of determination).
    R² = 1 - (SS_res / SS_tot)
    
    R² = 1.0 means perfect prediction
    R² = 0.0 means model is as good as predicting the mean
    R² < 0.0 means model is worse than predicting the mean
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
