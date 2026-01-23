"""
Keras Callbacks for Automatic TensorBoard Tracking

Provides comprehensive callbacks for logging various aspects of training
to TensorBoard and Vertex AI Experiments.
"""

from typing import TYPE_CHECKING, Optional
import tensorflow as tf
import numpy as np

if TYPE_CHECKING:
    from .tracker import ExperimentTracker


class ScalarCallback(tf.keras.callbacks.Callback):
    """
    Logs scalar metrics at the end of each epoch.
    
    Automatically captures all metrics from Keras logs including:
    - Training loss, validation loss
    - Custom metrics (RMSE, R², MAE, etc.)
    - Learning rate tracking
    
    Metrics are logged to both TensorBoard and Vertex AI Experiments.
    """
    
    def __init__(self, tracker: "ExperimentTracker"):
        """
        Args:
            tracker: ExperimentTracker instance for logging
        """
        super().__init__()
        self.tracker = tracker
    
    def on_epoch_end(self, epoch, logs=None):
        """Log all metrics at epoch end."""
        if logs is None:
            return
        
        # Log all metrics from Keras
        for name, value in logs.items():
            # Organize metrics by train/val split
            if name.startswith('val_'):
                tb_name = f"epoch/validation/{name[4:]}"
            else:
                tb_name = f"epoch/training/{name}"
            
            self.tracker.log_scalar(tb_name, float(value), step=epoch)
        
        # Log learning rate if available
        if hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'learning_rate'):
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                lr_value = float(lr.numpy())
            elif callable(lr):
                lr_value = float(lr(self.model.optimizer.iterations))
            else:
                lr_value = float(lr)
            self.tracker.log_scalar("epoch/learning_rate", lr_value, step=epoch)
        
        # Flush after each epoch to ensure data is persisted
        self.tracker.flush()


class HistogramCallback(tf.keras.callbacks.Callback):
    """
    Logs weight and gradient distributions to TensorBoard.
    
    Enables the Histograms and Distributions tabs in TensorBoard, which are
    essential for debugging training dynamics:
    - Vanishing gradients (distributions collapsing to 0)
    - Exploding gradients (distributions spreading wide)
    - Dead neurons (weights stuck at 0)
    - Learning progress (weight evolution over time)
    """
    
    def __init__(
        self,
        tracker: "ExperimentTracker",
        freq: int = 1,
        include_gradients: bool = False
    ):
        """
        Args:
            tracker: ExperimentTracker instance for logging
            freq: Log histograms every N epochs
            include_gradients: Also log gradient distributions (requires tape, more expensive)
        """
        super().__init__()
        self.tracker = tracker
        self.freq = freq
        self.include_gradients = include_gradients
    
    def on_epoch_end(self, epoch, logs=None):
        """Log weight histograms at epoch end."""
        if epoch % self.freq != 0:
            return
        
        # Log weight distributions for all layers
        for layer in self.model.layers:
            if not layer.weights:
                continue
            
            for weight in layer.weights:
                # Create clean name for TensorBoard
                weight_name = weight.name.replace(':', '_').replace('/', '_')
                name = f"weights/{layer.name}/{weight_name}"
                
                # Log weight distribution
                self.tracker.log_histogram(name, weight.numpy(), step=epoch)
        
        self.tracker.flush()
    
    def on_train_begin(self, logs=None):
        """Log initial weight distributions."""
        self.on_epoch_end(epoch=0, logs=logs)


class GraphCallback(tf.keras.callbacks.Callback):
    """
    Logs the model architecture graph to TensorBoard.
    
    This enables the Graph tab in TensorBoard, showing:
    - Model computational flow
    - Layer connections
    - Tensor shapes
    - Operation types
    """
    
    def __init__(self, tracker: "ExperimentTracker"):
        """
        Args:
            tracker: ExperimentTracker instance for logging
        """
        super().__init__()
        self.tracker = tracker
        self._logged = False
    
    def on_train_begin(self, logs=None):
        """Log model graph once at training start."""
        if not self._logged:
            self.tracker.log_graph(self.model)
            self._logged = True


class HParamsCallback(tf.keras.callbacks.Callback):
    """
    Logs hyperparameters and final metrics for TensorBoard HParams plugin.
    
    This enables the HParams tab in TensorBoard for hyperparameter comparison
    across multiple runs. Essential for experiment tracking and optimization.
    """
    
    def __init__(self, tracker: "ExperimentTracker"):
        """
        Args:
            tracker: ExperimentTracker instance for logging
        """
        super().__init__()
        self.tracker = tracker
    
    def on_train_end(self, logs=None):
        """Log final metrics associated with hyperparameters."""
        if logs is None:
            return
        
        # Extract final metrics
        final_metrics = {}
        for name, value in logs.items():
            if name.startswith('val_'):
                final_metrics[f"final_{name}"] = float(value)
        
        # Log to HParams plugin
        if final_metrics:
            self.tracker.log_hparams_metrics(final_metrics)
        
        self.tracker.flush()


class ProfilerCallback(tf.keras.callbacks.Callback):
    """
    Profiles GPU/CPU performance during training.
    
    Enables the Profile tab in TensorBoard showing:
    - GPU utilization
    - Memory usage
    - Operation timing
    - Bottleneck analysis
    
    Note: Profiling adds overhead, use sparingly (e.g., for a few batches).
    """
    
    def __init__(
        self,
        tracker: "ExperimentTracker",
        batch_range: tuple = (10, 20)
    ):
        """
        Args:
            tracker: ExperimentTracker instance for logging
            batch_range: (start_batch, end_batch) range to profile
        """
        super().__init__()
        self.tracker = tracker
        self.batch_range = batch_range
        self.start_batch, self.end_batch = batch_range
        self._profiling = False
    
    def on_train_batch_begin(self, batch, logs=None):
        """Start profiling at specified batch."""
        if batch == self.start_batch and not self._profiling:
            self.tracker.start_profiling()
            self._profiling = True
    
    def on_train_batch_end(self, batch, logs=None):
        """Stop profiling at specified batch."""
        if batch == self.end_batch and self._profiling:
            self.tracker.stop_profiling()
            self._profiling = False


class ImageCallback(tf.keras.callbacks.Callback):
    """
    Base class for custom image logging callbacks.
    
    Override this to create task-specific visualization callbacks:
    - Prediction comparisons (actual vs predicted)
    - Attention maps
    - Feature visualizations
    - Error distributions
    
    Example:
        class PredictionVisualizationCallback(ImageCallback):
            def __init__(self, tracker, val_data, freq=5):
                super().__init__(tracker, freq)
                self.val_data = val_data
            
            def create_visualization(self, epoch):
                # Create custom visualization
                fig, ax = plt.subplots()
                # ... plotting code ...
                return fig
    """
    
    def __init__(self, tracker: "ExperimentTracker", freq: int = 5):
        """
        Args:
            tracker: ExperimentTracker instance for logging
            freq: Create visualizations every N epochs
        """
        super().__init__()
        self.tracker = tracker
        self.freq = freq
    
    def on_epoch_end(self, epoch, logs=None):
        """Generate and log visualization at epoch end."""
        if epoch % self.freq != 0:
            return
        
        try:
            image = self.create_visualization(epoch)
            if image is not None:
                self.tracker.log_image("predictions/comparison", image, step=epoch)
                self.tracker.flush()
        except Exception as e:
            print(f"Warning: Failed to create visualization: {e}")
    
    def create_visualization(self, epoch: int) -> Optional[np.ndarray]:
        """
        Override this method to create custom visualizations.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Image array (H, W, C) or None to skip logging
        """
        raise NotImplementedError("Override create_visualization() in subclass")
