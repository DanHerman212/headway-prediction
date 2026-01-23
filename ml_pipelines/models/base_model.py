"""
Base Model Class

Abstract base class for all model architectures. Provides common interface
and utilities for model creation, compilation, and management.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import tensorflow as tf
from ml_pipelines.config import ModelConfig


class BaseModel(ABC):
    """
    Abstract base class for all model architectures.
    
    Provides:
    - Common interface for model creation
    - Configuration management
    - Model compilation with standard settings
    - Model summary and visualization
    
    Subclasses must implement:
    - build() method to create the model architecture
    
    Example:
        class MyModel(BaseModel):
            def build(self) -> tf.keras.Model:
                inputs = tf.keras.Input(shape=self.config.input_shape)
                x = tf.keras.layers.Dense(128, activation='relu')(inputs)
                outputs = tf.keras.layers.Dense(1)(x)
                return tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Usage
        model_instance = MyModel(config)
        model = model_instance.create()
        model.compile(...)
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize base model.
        
        Args:
            config: ModelConfig instance with architecture parameters
        """
        self.config = config
        self._model: Optional[tf.keras.Model] = None
    
    @abstractmethod
    def build(self) -> tf.keras.Model:
        """
        Build the model architecture.
        
        Subclasses must implement this method to define their specific
        architecture using Keras layers.
        
        Returns:
            Uncompiled Keras Model instance
        """
        pass
    
    def create(self) -> tf.keras.Model:
        """
        Create the model and cache it.
        
        Returns:
            Built Keras Model instance
        """
        if self._model is None:
            self._model = self.build()
            print(f"✓ Model created: {self.__class__.__name__}")
            print(f"  Total params: {self._model.count_params():,}")
        
        return self._model
    
    def get_model(self) -> Optional[tf.keras.Model]:
        """
        Get the cached model instance.
        
        Returns:
            Keras Model instance or None if not yet created
        """
        return self._model
    
    def summary(self):
        """Print model summary."""
        if self._model is None:
            self.create()
        self._model.summary()
    
    def plot_model(self, save_path: str = "model_architecture.png", **kwargs):
        """
        Plot model architecture diagram.
        
        Args:
            save_path: Path to save the diagram
            **kwargs: Additional arguments for tf.keras.utils.plot_model
        """
        if self._model is None:
            self.create()
        
        tf.keras.utils.plot_model(
            self._model,
            to_file=save_path,
            show_shapes=True,
            show_layer_names=True,
            **kwargs
        )
        print(f"✓ Model architecture saved to: {save_path}")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get model configuration as dictionary.
        
        Returns:
            Dictionary of configuration parameters
        """
        return self.config.to_dict()
    
    def save_config(self, path: str):
        """
        Save model configuration to YAML file.
        
        Args:
            path: Output path for configuration file
        """
        self.config.save_yaml(path)
        print(f"✓ Model configuration saved to: {path}")
