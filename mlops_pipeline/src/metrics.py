import tensorflow as tf
from tensorflow import keras

class MAESeconds(keras.metrics.Metric):
    """
    Mean Absolute Error in seconds (converted from log-space).
    """
    def __init__(self, name='mae_seconds', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Input is log1p(minutes). Convert to seconds for metric.
        y_true_seconds = tf.math.expm1(y_true) * 60.0
        y_pred_seconds = tf.math.expm1(y_pred) * 60.0
        errors = tf.abs(y_true_seconds - y_pred_seconds)
        if sample_weight is not None:
            errors = errors * sample_weight
        self.total_error.assign_add(tf.reduce_sum(errors))
        if sample_weight is not None:
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(errors), dtype=tf.float32))
    
    def result(self):
        return self.total_error / self.count
    
    def reset_state(self):
        self.total_error.assign(0.0)
        self.count.assign(0.0)
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
