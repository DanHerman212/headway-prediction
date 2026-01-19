"""
Core Tracker class for TensorBoard logging.
"""

import os
from typing import Dict, Any, List, Union, Optional
from datetime import datetime

import tensorflow as tf
import numpy as np

from .config import TrackerConfig
from .callbacks import (
    ScalarCallback,
    HistogramCallback,
    HParamsCallback,
    ProfilerCallback,
    GraphCallback,
)


class Tracker:
    """
    Universal TensorBoard tracking for TensorFlow/Keras experiments.
    
    Provides a unified interface for logging all TensorBoard data types:
    - Scalars (loss, metrics, learning rate)
    - Histograms (weight distributions, gradients)
    - Graphs (model architecture)
    - HParams (hyperparameter comparison)
    - Text (configuration, notes)
    - Profiling (GPU/CPU performance)
    
    Example:
        tracker = Tracker(TrackerConfig(
            experiment_name="my-experiment",
            run_name="run-001",
            log_dir="gs://my-bucket/tensorboard/run-001",
            histograms=True,
            hparams_dict={"lr": 0.001, "batch_size": 32}
        ))
        
        # Log model graph
        tracker.log_graph(model)
        
        # Use with Keras
        model.fit(x, y, callbacks=tracker.keras_callbacks())
        
        # Or manual logging
        tracker.log_scalar("custom/metric", 0.5, step=100)
        tracker.log_histogram("layer1/weights", weights, step=100)
        
        tracker.close()
    
    Context manager usage:
        with Tracker(config) as tracker:
            model.fit(x, y, callbacks=tracker.keras_callbacks())
    """
    
    def __init__(self, config: Union[TrackerConfig, Dict, str], use_vertex_experiments: bool = True):
        """
        Initialize the Tracker.
        
        Args:
            config: TrackerConfig object, dict, or path to YAML file
            use_vertex_experiments: Whether to log to Vertex AI Experiments (default: True)
        """
        # Parse config
        if isinstance(config, str):
            config = TrackerConfig.from_yaml(config)
        elif isinstance(config, dict):
            config = TrackerConfig.from_dict(config)
        
        self.config = config
        self.use_vertex_experiments = use_vertex_experiments
        self._vertex_run = None
        
        self._init_writer()
        self._init_vertex_experiments()
        self._log_initial_info()
    
    def _init_writer(self):
        """Initialize TensorFlow summary writer."""
        self.log_dir = self.config.log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)
        print(f"TensorBoard logging to: {self.log_dir}")
    
    def _init_vertex_experiments(self):
        """Initialize Vertex AI Experiments tracking."""
        if not self.use_vertex_experiments:
            return
            
        try:
            from google.cloud import aiplatform
            
            # Initialize Vertex AI if not already done
            aiplatform.init()
            
            # Get or create experiment
            experiment = aiplatform.Experiment.get_or_create(
                experiment_name=self.config.experiment_name,
                description=self.config.description or f"Experiment: {self.config.experiment_name}",
            )
            
            # Start a run within the experiment
            self._vertex_run = aiplatform.start_run(
                run=self.config.run_name,
                experiment=self.config.experiment_name,
                tensorboard=self.log_dir,
            )
            
            # Log hyperparameters to Vertex AI
            if self.config.hparams_dict:
                aiplatform.log_params(self.config.hparams_dict)
            
            print(f"Vertex AI Experiments: {self.config.experiment_name}/{self.config.run_name}")
            
        except ImportError:
            print("Warning: google-cloud-aiplatform not available, skipping Vertex AI Experiments")
            self.use_vertex_experiments = False
        except Exception as e:
            print(f"Warning: Failed to initialize Vertex AI Experiments: {e}")
            self.use_vertex_experiments = False
    
    def _log_initial_info(self):
        """Log initial experiment information."""
        # Log config as text
        config_text = f"""
Experiment: {self.config.experiment_name}
Run: {self.config.run_name}
Started: {datetime.now().isoformat()}
Description: {self.config.description or 'N/A'}

Tracking Configuration:
- Scalars: {self.config.scalars}
- Histograms: {self.config.histograms} (freq={self.config.histogram_freq})
- Graphs: {self.config.graphs}
- HParams: {self.config.hparams}
- Profiling: {self.config.profiling}
"""
        self.log_text("config/experiment", config_text, step=0)
        
        # Log hyperparameters as text
        if self.config.hparams_dict:
            hparams_text = "\n".join(
                f"- {k}: {v}" for k, v in self.config.hparams_dict.items()
            )
            self.log_text("config/hyperparameters", hparams_text, step=0)
        
        # Log HParams for TensorBoard HParams plugin
        if self.config.hparams and self.config.hparams_dict:
            self._log_hparams_plugin()
    
    def _log_hparams_plugin(self):
        """Log hyperparameters for TensorBoard HParams plugin."""
        try:
            from tensorboard.plugins.hparams import api as hp
            
            with self.writer.as_default():
                hp.hparams(self.config.hparams_dict)
        except ImportError:
            print("Warning: tensorboard.plugins.hparams not available")
        except Exception as e:
            print(f"Warning: Failed to log hparams: {e}")
    
    # =========================================================================
    # Core Logging Methods
    # =========================================================================
    
    def log_scalar(self, name: str, value: float, step: int):
        """
        Log a scalar metric to both TensorBoard and Vertex AI Experiments.
        
        Args:
            name: Metric name (e.g., "train/loss", "val/accuracy")
            value: Scalar value
            step: Training step or epoch
        """
        # Log to TensorBoard
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=step)
        
        # Log to Vertex AI Experiments
        if self.use_vertex_experiments and self._vertex_run:
            try:
                from google.cloud import aiplatform
                # Convert name to valid metric key (replace / with _)
                metric_key = name.replace("/", "_")
                aiplatform.log_metrics({metric_key: float(value)})
            except Exception as e:
                pass  # Silently fail for Vertex AI logging
    
    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log multiple scalar metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Training step or epoch
            prefix: Optional prefix for all metric names
        """
        with self.writer.as_default():
            for name, value in metrics.items():
                full_name = f"{prefix}/{name}" if prefix else name
                tf.summary.scalar(full_name, value, step=step)
    
    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """
        Log a histogram (weight distributions, gradients, etc.).
        
        Args:
            name: Histogram name (e.g., "weights/dense_1/kernel")
            values: Array of values
            step: Training step or epoch
        """
        with self.writer.as_default():
            tf.summary.histogram(name, values, step=step)
    
    def log_image(self, name: str, image: np.ndarray, step: int):
        """
        Log an image.
        
        Args:
            name: Image name
            image: Image array with shape (H, W, C), (H, W), (B, H, W, C), or (B, H, W)
            step: Training step or epoch
        """
        # Ensure correct shape
        if image.ndim == 2:
            image = image[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)
        elif image.ndim == 3:
            if image.shape[-1] not in [1, 3, 4]:
                # Assume (H, W, C) but C is wrong, treat as grayscale batch
                image = image[:, :, :, np.newaxis]
            else:
                image = image[np.newaxis, :, :, :]  # (1, H, W, C)
        
        with self.writer.as_default():
            tf.summary.image(name, image, step=step)
    
    def log_images(self, name: str, images: np.ndarray, step: int, max_outputs: int = 4):
        """
        Log multiple images.
        
        Args:
            name: Image name
            images: Batch of images (B, H, W, C)
            step: Training step or epoch
            max_outputs: Maximum number of images to display
        """
        with self.writer.as_default():
            tf.summary.image(name, images, step=step, max_outputs=max_outputs)
    
    def log_text(self, name: str, text: str, step: int = 0):
        """
        Log text (configuration, notes, error messages).
        
        Args:
            name: Text identifier
            text: Text content
            step: Training step or epoch
        """
        with self.writer.as_default():
            tf.summary.text(name, text, step=step)
    
    def log_graph(self, model: tf.keras.Model):
        """
        Log model architecture graph.
        
        Args:
            model: Keras model to log
        """
        try:
            @tf.function
            def trace_fn(x):
                return model(x, training=False)
            
            # Get input shape from model
            if hasattr(model, 'input_shape'):
                input_shape = model.input_shape
            elif hasattr(model, 'inputs') and model.inputs:
                input_shape = [inp.shape for inp in model.inputs]
            else:
                print("Warning: Could not determine model input shape for graph logging")
                return
            
            # Create dummy input
            if isinstance(input_shape, list):
                # Multi-input model
                dummy_inputs = []
                for shape in input_shape:
                    concrete_shape = [1 if dim is None else dim for dim in shape]
                    dummy_inputs.append(tf.zeros(concrete_shape))
                dummy_input = dummy_inputs
            else:
                concrete_shape = [1 if dim is None else dim for dim in input_shape]
                dummy_input = tf.zeros(concrete_shape)
            
            with self.writer.as_default():
                tf.summary.trace_on(graph=True, profiler=False)
                if isinstance(dummy_input, list):
                    _ = model(dummy_input, training=False)
                else:
                    _ = model(dummy_input, training=False)
                tf.summary.trace_export(name="model_graph", step=0)
            
            print("Model graph logged to TensorBoard")
            
        except Exception as e:
            print(f"Warning: Failed to log model graph: {e}")
    
    def log_hparams_metrics(self, metrics: Dict[str, float]):
        """
        Log final metrics associated with hyperparameters.
        
        Call this at the end of training to associate final metrics
        with the hyperparameters for the HParams comparison view.
        
        Args:
            metrics: Final metrics (e.g., {"best_val_loss": 0.01})
        """
        try:
            from tensorboard.plugins.hparams import api as hp
            
            with self.writer.as_default():
                for name, value in metrics.items():
                    tf.summary.scalar(name, value, step=1)
        except Exception as e:
            print(f"Warning: Failed to log hparams metrics: {e}")
    
    # =========================================================================
    # Profiling
    # =========================================================================
    
    def start_profiling(self):
        """Start GPU/CPU profiling."""
        if self.config.profiling:
            try:
                tf.profiler.experimental.start(self.log_dir)
                print(f"Profiling started, logging to {self.log_dir}")
            except Exception as e:
                print(f"Warning: Failed to start profiling: {e}")
    
    def stop_profiling(self):
        """Stop GPU/CPU profiling."""
        if self.config.profiling:
            try:
                tf.profiler.experimental.stop()
                print("Profiling stopped")
            except Exception as e:
                print(f"Warning: Failed to stop profiling: {e}")
    
    # =========================================================================
    # Keras Integration
    # =========================================================================
    
    def keras_callbacks(
        self,
        validation_data: Optional[tf.data.Dataset] = None,
        custom_image_callback: Optional[tf.keras.callbacks.Callback] = None,
    ) -> List[tf.keras.callbacks.Callback]:
        """
        Get Keras callbacks for automatic tracking during model.fit().
        
        Returns all configured callbacks based on TrackerConfig settings.
        
        Args:
            validation_data: Optional validation dataset for image callbacks
            custom_image_callback: Optional custom image visualization callback
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Scalar metrics (always included if scalars=True)
        if self.config.scalars:
            callbacks.append(ScalarCallback(self))
        
        # Weight histograms
        if self.config.histograms:
            callbacks.append(HistogramCallback(
                self,
                freq=self.config.histogram_freq
            ))
        
        # Model graph (logs once at training start)
        if self.config.graphs:
            callbacks.append(GraphCallback(self))
        
        # HParams (logs at training end with final metrics)
        if self.config.hparams:
            callbacks.append(HParamsCallback(self))
        
        # Profiling
        if self.config.profiling:
            callbacks.append(ProfilerCallback(
                self,
                batch_range=self.config.profile_batch_range
            ))
        
        # Custom image callback
        if custom_image_callback:
            callbacks.append(custom_image_callback)
        
        return callbacks
    
    # =========================================================================
    # Context Manager
    # =========================================================================
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def close(self):
        """Flush and close the summary writer and Vertex AI run."""
        self.flush()
        self.writer.close()
        
        # End Vertex AI Experiments run
        if self.use_vertex_experiments and self._vertex_run:
            try:
                from google.cloud import aiplatform
                aiplatform.end_run()
                print(f"Vertex AI Experiment run ended: {self.config.experiment_name}/{self.config.run_name}")
            except Exception as e:
                print(f"Warning: Failed to end Vertex AI run: {e}")
        
        print(f"TensorBoard writer closed. View with:\n  tensorboard --logdir={self.log_dir}")
    
    def flush(self):
        """Flush pending writes to disk."""
        self.writer.flush()
