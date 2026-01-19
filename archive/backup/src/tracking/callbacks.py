"""
Keras callbacks for automatic TensorBoard tracking.
"""

from typing import TYPE_CHECKING, Tuple, Optional
import tensorflow as tf

if TYPE_CHECKING:
    from .tracker import Tracker


class ScalarCallback(tf.keras.callbacks.Callback):
    """
    Logs scalar metrics at end of each epoch.
    
    Automatically captures all metrics from Keras logs including:
    - loss, val_loss
    - Custom metrics (rmse_seconds, r_squared, etc.)
    - Learning rate
    """
    
    def __init__(self, tracker: "Tracker"):
        """
        Args:
            tracker: Tracker instance for logging
        """
        super().__init__()
        self.tracker = tracker
    
    def on_epoch_end(self, epoch, logs=None):
        """Log all metrics at epoch end."""
        if logs is None:
            return
        
        # Log all metrics from Keras
        for name, value in logs.items():
            # Clean up metric names for TensorBoard
            if name.startswith('val_'):
                tb_name = f"epoch/val/{name[4:]}"
            else:
                tb_name = f"epoch/train/{name}"
            
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
        
        # Flush after each epoch
        self.tracker.flush()


class HistogramCallback(tf.keras.callbacks.Callback):
    """
    Logs weight and gradient distributions.
    
    Enables the Histograms and Distributions tabs in TensorBoard.
    Useful for debugging:
    - Vanishing gradients (distributions collapsing to 0)
    - Exploding gradients (distributions spreading wide)
    - Dead neurons (weights stuck at 0)
    """
    
    def __init__(self, tracker: "Tracker", freq: int = 1, include_gradients: bool = False):
        """
        Args:
            tracker: Tracker instance for logging
            freq: Log every N epochs
            include_gradients: Also log gradient distributions (more expensive)
        """
        super().__init__()
        self.tracker = tracker
        self.freq = freq
        self.include_gradients = include_gradients
    
    def on_epoch_end(self, epoch, logs=None):
        """Log weight histograms at epoch end."""
        if epoch % self.freq != 0:
            return
        
        for layer in self.model.layers:
            # Skip layers without weights
            if not layer.weights:
                continue
            
            for weight in layer.weights:
                # Create clean name
                weight_name = weight.name.replace(':', '_').replace('/', '_')
                name = f"weights/{layer.name}/{weight_name}"
                
                # Log weight distribution
                self.tracker.log_histogram(name, weight.numpy(), step=epoch)
        
        self.tracker.flush()


class HParamsCallback(tf.keras.callbacks.Callback):
    """
    Logs final metrics associated with hyperparameters.
    
    Enables the HParams tab in TensorBoard for comparing runs
    with different hyperparameter configurations.
    """
    
    def __init__(self, tracker: "Tracker"):
        """
        Args:
            tracker: Tracker instance for logging
        """
        super().__init__()
        self.tracker = tracker
    
    def on_train_end(self, logs=None):
        """Log final metrics at training end."""
        if logs is None:
            return
        
        # Collect final metrics
        final_metrics = {}
        
        # Get best values from history if available
        if hasattr(self.model, 'history') and self.model.history:
            history = self.model.history.history
            
            if 'val_loss' in history:
                final_metrics['best_val_loss'] = min(history['val_loss'])
            if 'val_rmse_seconds' in history:
                final_metrics['best_val_rmse_seconds'] = min(history['val_rmse_seconds'])
            if 'val_r_squared' in history:
                final_metrics['best_val_r_squared'] = max(history['val_r_squared'])
            
            final_metrics['total_epochs'] = len(history.get('loss', []))
        
        # Log to HParams plugin
        if final_metrics:
            self.tracker.log_hparams_metrics(final_metrics)


class GraphCallback(tf.keras.callbacks.Callback):
    """
    Logs model architecture graph.
    
    Enables the Graphs tab in TensorBoard to visualize
    the model's computational graph.
    """
    
    def __init__(self, tracker: "Tracker"):
        """
        Args:
            tracker: Tracker instance for logging
        """
        super().__init__()
        self.tracker = tracker
        self._logged = False
    
    def on_train_begin(self, logs=None):
        """Log model graph at training start."""
        if self._logged:
            return
        
        self.tracker.log_graph(self.model)
        self._logged = True


class ProfilerCallback(tf.keras.callbacks.Callback):
    """
    Profiles GPU/CPU performance.
    
    Enables the Profile tab in TensorBoard to analyze:
    - GPU utilization
    - Memory usage
    - Operation timing
    - Bottleneck identification
    """
    
    def __init__(self, tracker: "Tracker", batch_range: Tuple[int, int] = (10, 20)):
        """
        Args:
            tracker: Tracker instance for logging
            batch_range: (start_batch, end_batch) to profile
        """
        super().__init__()
        self.tracker = tracker
        self.start_batch, self.stop_batch = batch_range
        self._profiling = False
    
    def on_batch_begin(self, batch, logs=None):
        """Start profiling at start_batch."""
        if batch == self.start_batch and not self._profiling:
            self.tracker.start_profiling()
            self._profiling = True
    
    def on_batch_end(self, batch, logs=None):
        """Stop profiling at stop_batch."""
        if batch == self.stop_batch and self._profiling:
            self.tracker.stop_profiling()
            self._profiling = False
    
    def on_train_end(self, logs=None):
        """Ensure profiling is stopped."""
        if self._profiling:
            self.tracker.stop_profiling()
            self._profiling = False


class ImageCallback(tf.keras.callbacks.Callback):
    """
    Base class for image visualization callbacks.
    
    Subclass this to create project-specific visualizations.
    See src/visualizations/ for examples.
    """
    
    def __init__(
        self,
        tracker: "Tracker",
        validation_data: Optional[tf.data.Dataset] = None,
        freq: int = 5,
        num_samples: int = 4,
    ):
        """
        Args:
            tracker: Tracker instance for logging
            validation_data: Validation dataset for generating predictions
            freq: Log every N epochs
            num_samples: Number of samples to visualize
        """
        super().__init__()
        self.tracker = tracker
        self.validation_data = validation_data
        self.freq = freq
        self.num_samples = num_samples
    
    def on_epoch_end(self, epoch, logs=None):
        """Generate and log visualization at epoch end."""
        if epoch % self.freq != 0:
            return
        
        if self.validation_data is None:
            return
        
        # Get sample batch
        for x, y_true in self.validation_data.take(1):
            y_pred = self.model.predict(x[:self.num_samples], verbose=0)
            y_true = y_true[:self.num_samples]
        
        # Generate visualization (override in subclass)
        image = self.create_visualization(y_true, y_pred, epoch)
        
        if image is not None:
            self.tracker.log_image("predictions", image, step=epoch)
    
    def create_visualization(self, y_true, y_pred, epoch: int):
        """
        Create visualization image. Override in subclass.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            epoch: Current epoch
            
        Returns:
            Image as numpy array (B, H, W, C) or None
        """
        raise NotImplementedError("Subclass must implement create_visualization()")
