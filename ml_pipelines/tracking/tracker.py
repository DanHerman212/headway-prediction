"""
Experiment Tracker

Unified interface for logging to both TensorBoard and Vertex AI Experiments.
This is the core tracking class that integrates all experiment data.
"""

import os
from typing import Dict, Any, List, Union, Optional
from datetime import datetime

import tensorflow as tf
import numpy as np

from ml_pipelines.config import TrackingConfig
from .callbacks import (
    ScalarCallback,
    HistogramCallback,
    GraphCallback,
    HParamsCallback,
    ProfilerCallback,
)


class ExperimentTracker:
    """
    Universal experiment tracking for TensorFlow/Keras with full TensorBoard
    and Vertex AI Experiments integration.
    
    Provides a unified interface for logging all experiment data:
    - Scalars (loss, metrics, learning rate, custom metrics)
    - Histograms (weight distributions, gradients)
    - Graphs (model architecture)
    - HParams (hyperparameter comparison)
    - Text (configuration, notes, errors)
    - Images (predictions, visualizations)
    - Profiling (GPU/CPU performance)
    
    All data is automatically logged to both TensorBoard (for detailed analysis)
    and Vertex AI Experiments (for experiment management and comparison).
    
    Example Usage:
        # Create configuration
        tracking_config = TrackingConfig.create_from_model_config(
            model_config=model_config,
            experiment_name="headway-prediction",
            run_name="baseline-001",
            vertex_project="my-project",
        )
        
        # Initialize tracker
        tracker = ExperimentTracker(tracking_config)
        
        # Log model graph
        tracker.log_graph(model)
        
        # Use with Keras training
        model.fit(
            train_data,
            validation_data=val_data,
            callbacks=tracker.keras_callbacks()
        )
        
        # Manual logging
        tracker.log_scalar("custom/metric", 0.95, step=100)
        tracker.log_histogram("layer_activations", activations, step=100)
        
        # Close tracker
        tracker.close()
    
    Context Manager Usage:
        with ExperimentTracker(config) as tracker:
            model.fit(x, y, callbacks=tracker.keras_callbacks())
    """
    
    def __init__(
        self,
        config: Union[TrackingConfig, Dict, str],
        auto_init_vertex: bool = True
    ):
        """
        Initialize the ExperimentTracker.
        
        Args:
            config: TrackingConfig object, dict, or path to YAML file
            auto_init_vertex: Whether to automatically initialize Vertex AI
        """
        # Parse configuration
        if isinstance(config, str):
            config = TrackingConfig.from_yaml(config)
        elif isinstance(config, dict):
            config = TrackingConfig.from_dict(config)
        
        self.config = config
        self.config.validate()
        
        self._vertex_run = None
        self._vertex_initialized = False
        
        # Initialize TensorBoard writer
        self._init_writer()
        
        # Initialize Vertex AI Experiments
        if auto_init_vertex and self.config.use_vertex_experiments:
            self._init_vertex_experiments()
        
        # Log initial experiment information
        self._log_initial_info()
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def _init_writer(self):
        """Initialize TensorFlow summary writer for TensorBoard."""
        self.log_dir = self.config.log_dir
        
        # Create directory if it doesn't exist
        if not self.log_dir.startswith('gs://'):
            os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = tf.summary.create_file_writer(self.log_dir)
        print(f"✓ TensorBoard logging initialized: {self.log_dir}")
        print(f"  View with: {self.config.get_tensorboard_command()}")
    
    def _init_vertex_experiments(self):
        """Initialize Vertex AI Experiments tracking."""
        try:
            from google.cloud import aiplatform
            
            # Initialize Vertex AI
            aiplatform.init(
                project=self.config.vertex_project,
                location=self.config.vertex_location,
            )
            self._vertex_initialized = True
            
            # Get or create experiment
            try:
                experiment = aiplatform.Experiment(
                    experiment_name=self.config.experiment_name
                )
                print(f"✓ Using existing Vertex AI Experiment: {self.config.experiment_name}")
            except:
                experiment = aiplatform.Experiment.create(
                    experiment_name=self.config.experiment_name,
                    description=self.config.description or f"Experiment: {self.config.experiment_name}",
                )
                print(f"✓ Created new Vertex AI Experiment: {self.config.experiment_name}")
            
            # Start a run within the experiment
            aiplatform.start_run(
                run=self.config.run_name,
                tensorboard=self.config.tensorboard_resource_name,
            )
            self._vertex_run = aiplatform.get_experiment_run(self.config.run_name)
            
            # Log hyperparameters to Vertex AI
            if self.config.hparams_dict:
                aiplatform.log_params(self.config.hparams_dict)
            
            print(f"✓ Vertex AI run started: {self.config.experiment_name}/{self.config.run_name}")
            
        except ImportError:
            print("⚠ Warning: google-cloud-aiplatform not installed")
            print("  Install with: pip install google-cloud-aiplatform>=1.38.0")
            print("  Continuing with TensorBoard-only logging")
            self.config.use_vertex_experiments = False
            
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize Vertex AI Experiments: {e}")
            print("  Continuing with TensorBoard-only logging")
            self.config.use_vertex_experiments = False
    
    def _log_initial_info(self):
        """Log initial experiment information to TensorBoard."""
        # Create experiment summary text
        config_text = f"""
# Experiment: {self.config.experiment_name}
**Run:** {self.config.run_name}  
**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Description:** {self.config.description or 'N/A'}

## Tracking Configuration
- **Scalars:** {self.config.scalars}
- **Histograms:** {self.config.histograms} (every {self.config.histogram_freq} epochs)
- **Graphs:** {self.config.graphs}
- **HParams:** {self.config.hparams}
- **Profiling:** {self.config.profiling}
- **Vertex AI:** {self.config.use_vertex_experiments}

## Hyperparameters
{self._format_hparams()}
"""
        self.log_text("experiment/config", config_text, step=0)
        
        # Log HParams for TensorBoard HParams plugin
        if self.config.hparams and self.config.hparams_dict:
            self._log_hparams_plugin()
    
    def _format_hparams(self) -> str:
        """Format hyperparameters as markdown list."""
        if not self.config.hparams_dict:
            return "No hyperparameters specified"
        
        lines = []
        for key, value in self.config.hparams_dict.items():
            lines.append(f"- **{key}:** {value}")
        return "\n".join(lines)
    
    def _log_hparams_plugin(self):
        """Log hyperparameters for TensorBoard HParams plugin."""
        try:
            from tensorboard.plugins.hparams import api as hp
            
            with self.writer.as_default():
                hp.hparams(self.config.hparams_dict)
                
        except ImportError:
            pass  # TensorBoard HParams plugin not available
        except Exception as e:
            print(f"⚠ Warning: Failed to log hparams: {e}")
    
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
        if self.config.use_vertex_experiments and self._vertex_run:
            try:
                from google.cloud import aiplatform
                # Convert name to valid metric key (replace / with _)
                metric_key = name.replace("/", "_").replace("-", "_")
                aiplatform.log_metrics({metric_key: float(value)})
            except Exception:
                pass  # Silently fail for Vertex AI logging
    
    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log multiple scalar metrics at once.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Training step or epoch
            prefix: Optional prefix for all metric names (e.g., "train", "val")
        """
        for name, value in metrics.items():
            full_name = f"{prefix}/{name}" if prefix else name
            self.log_scalar(full_name, value, step)
    
    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """
        Log a histogram (weight distributions, gradients, activations).
        
        Args:
            name: Histogram name (e.g., "weights/dense_1/kernel")
            values: Array of values
            step: Training step or epoch
        """
        with self.writer.as_default():
            tf.summary.histogram(name, values, step=step)
    
    def log_image(self, name: str, image: np.ndarray, step: int):
        """
        Log a single image to TensorBoard.
        
        Args:
            name: Image name (e.g., "predictions/sample_1")
            image: Image array with shape (H, W, C), (H, W), (B, H, W, C), or (B, H, W)
            step: Training step or epoch
        """
        # Normalize image shape for TensorBoard
        if image.ndim == 2:
            # Grayscale (H, W) -> (1, H, W, 1)
            image = image[np.newaxis, :, :, np.newaxis]
        elif image.ndim == 3:
            if image.shape[-1] not in [1, 3, 4]:
                # Treat as batch of grayscale images
                image = image[:, :, :, np.newaxis]
            else:
                # Single RGB/RGBA image (H, W, C) -> (1, H, W, C)
                image = image[np.newaxis, :, :, :]
        
        with self.writer.as_default():
            tf.summary.image(name, image, step=step, max_outputs=1)
    
    def log_images(self, name: str, images: np.ndarray, step: int, max_outputs: int = 4):
        """
        Log multiple images to TensorBoard.
        
        Args:
            name: Base name for images
            images: Batch of images (B, H, W, C) or (B, H, W)
            step: Training step or epoch
            max_outputs: Maximum number of images to display
        """
        # Ensure 4D shape
        if images.ndim == 3:
            images = images[:, :, :, np.newaxis]
        
        with self.writer.as_default():
            tf.summary.image(name, images, step=step, max_outputs=max_outputs)
    
    def log_text(self, name: str, text: str, step: int = 0):
        """
        Log text information (configuration, notes, errors).
        
        Args:
            name: Text identifier (e.g., "config/model", "notes/epoch_5")
            text: Text content (supports markdown)
            step: Training step or epoch
        """
        with self.writer.as_default():
            tf.summary.text(name, text, step=step)
    
    def log_graph(self, model: tf.keras.Model):
        """
        Log model architecture graph to TensorBoard.
        
        Enables the Graph tab showing model structure, tensor shapes,
        and computational flow.
        
        Args:
            model: Keras model to log
        """
        try:
            # Get model input shape
            if hasattr(model, 'input_shape'):
                input_shape = model.input_shape
            elif hasattr(model, 'inputs') and model.inputs:
                input_shape = [inp.shape for inp in model.inputs]
            else:
                print("⚠ Warning: Could not determine model input shape for graph logging")
                return
            
            # Create dummy input
            if isinstance(input_shape, list):
                # Multi-input model
                dummy_inputs = []
                for shape in input_shape:
                    concrete_shape = [1 if dim is None else dim for dim in shape]
                    dummy_inputs.append(tf.zeros(concrete_shape))
            else:
                # Single input model
                concrete_shape = [1 if dim is None else dim for dim in input_shape]
                dummy_inputs = tf.zeros(concrete_shape)
            
            # Trace model execution
            with self.writer.as_default():
                tf.summary.trace_on(graph=True, profiler=False)
                _ = model(dummy_inputs, training=False)
                tf.summary.trace_export(name="model_graph", step=0)
            
            print("✓ Model graph logged to TensorBoard")
            
        except Exception as e:
            print(f"⚠ Warning: Failed to log model graph: {e}")
    
    def log_hparams_metrics(self, metrics: Dict[str, float]):
        """
        Log final metrics for hyperparameter comparison.
        
        Call this at the end of training to associate final metrics
        with hyperparameters in the TensorBoard HParams tab.
        
        Args:
            metrics: Final metrics (e.g., {"best_val_loss": 0.01, "final_accuracy": 0.95})
        """
        try:
            from tensorboard.plugins.hparams import api as hp
            
            with self.writer.as_default():
                for name, value in metrics.items():
                    tf.summary.scalar(f"hparams/{name}", value, step=1)
                    
        except Exception as e:
            print(f"⚠ Warning: Failed to log hparams metrics: {e}")
    
    # =========================================================================
    # Profiling
    # =========================================================================
    
    def start_profiling(self):
        """Start GPU/CPU profiling for performance analysis."""
        if self.config.profiling:
            try:
                tf.profiler.experimental.start(self.log_dir)
                print(f"✓ Profiling started")
            except Exception as e:
                print(f"⚠ Warning: Failed to start profiling: {e}")
    
    def stop_profiling(self):
        """Stop GPU/CPU profiling."""
        if self.config.profiling:
            try:
                tf.profiler.experimental.stop()
                print("✓ Profiling stopped")
            except Exception as e:
                print(f"⚠ Warning: Failed to stop profiling: {e}")
    
    # =========================================================================
    # Keras Integration
    # =========================================================================
    
    def keras_callbacks(
        self,
        custom_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> List[tf.keras.callbacks.Callback]:
        """
        Get Keras callbacks for automatic tracking during model.fit().
        
        Returns all configured callbacks based on TrackingConfig settings.
        These callbacks automatically log scalars, histograms, graphs, and more.
        
        Args:
            custom_callbacks: Optional list of custom callbacks to include
            
        Returns:
            List of Keras callbacks ready for model.fit()
        """
        callbacks = []
        
        # Scalar metrics (loss, accuracy, etc.)
        if self.config.scalars:
            callbacks.append(ScalarCallback(self))
        
        # Weight and gradient histograms
        if self.config.histograms:
            callbacks.append(HistogramCallback(
                self,
                freq=self.config.histogram_freq
            ))
        
        # Model architecture graph
        if self.config.graphs:
            callbacks.append(GraphCallback(self))
        
        # Hyperparameter tracking
        if self.config.hparams:
            callbacks.append(HParamsCallback(self))
        
        # GPU/CPU profiling
        if self.config.profiling:
            callbacks.append(ProfilerCallback(
                self,
                batch_range=self.config.profile_batch_range
            ))
        
        # Add custom callbacks
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        
        return callbacks
    
    # =========================================================================
    # Context Manager
    # =========================================================================
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.close()
    
    def close(self):
        """Flush and close the tracker, ending the Vertex AI run."""
        self.flush()
        self.writer.close()
        
        # End Vertex AI Experiments run
        if self.config.use_vertex_experiments and self._vertex_run:
            try:
                from google.cloud import aiplatform
                aiplatform.end_run()
                print(f"✓ Vertex AI run ended: {self.config.experiment_name}/{self.config.run_name}")
            except Exception as e:
                print(f"⚠ Warning: Failed to end Vertex AI run: {e}")
        
        print(f"✓ Experiment tracking closed")
        print(f"  TensorBoard: {self.config.get_tensorboard_command()}")
    
    def flush(self):
        """Flush pending writes to disk/GCS."""
        self.writer.flush()
