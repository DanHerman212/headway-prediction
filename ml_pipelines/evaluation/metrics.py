"""
Custom Metrics for Headway Prediction

Since headway is stored as log(seconds), we need metrics that convert
predictions back to real seconds for interpretability.
"""

import tensorflow as tf
from tensorflow import keras

class MAESeconds(keras.metrics.Metric):
    """
    Mean Absolute Error in seconds (converted from log-space).
    
    Since headway is stored as log(seconds), this metric:
    1. Converts predictions from log-space to seconds: exp(log_headway)
    2. Calculates MAE in actual seconds
    
    This makes the metric interpretable: "Average error is X seconds"
    
    Example:
        model.compile(
            loss='huber',
            metrics={'headway': [MAESeconds(name='mae_seconds')]}
        )
    """
    
    def __init__(self, name='mae_seconds', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state with new batch."""
        # Convert from log-space to seconds
        y_true_seconds = tf.exp(y_true)
        y_pred_seconds = tf.exp(y_pred)
        
        # Calculate absolute error in seconds
        errors = tf.abs(y_true_seconds - y_pred_seconds)
        
        # Apply sample weights if provided
        if sample_weight is not None:
            errors = errors * sample_weight
        
        # Update running totals
        self.total_error.assign_add(tf.reduce_sum(errors))
        
        if sample_weight is not None:
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(errors), dtype=tf.float32))
    
    def result(self):
        """Return the mean absolute error in seconds."""
        return self.total_error / self.count
    
    def reset_state(self):
        """Reset metric state."""
        self.total_error.assign(0.0)
        self.count.assign(0.0)
