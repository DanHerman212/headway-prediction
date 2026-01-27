from tensorflow import keras
from google.cloud import aiplatform

class VertexAILoggingCallback(keras.callbacks.Callback):
    """Logs metrics to Vertex AI Experiments at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        try:
            if logs:
                # Cast all metrics to Python float to avoid Vertex AI TypeError with numpy types
                metrics = {k: float(v) for k, v in logs.items()}
                # Log both scalar metrics (for summary) and time-series (for plotting)
                aiplatform.log_metrics(metrics)
        except Exception as e:
            print(f"ERROR: Failed to log metrics to Vertex AI in callback: {e}")

        # Attempt Time Series Logging (Experimental)
        try:
            if logs:
                 metrics = {k: float(v) for k, v in logs.items()}
                 aiplatform.log_time_series_metrics(metrics, step=epoch + 1)
        except Exception as e:
            print(f"Warning: Failed to log time series metrics: {e}")
