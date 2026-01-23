"""
Model Builder Utility

Provides utilities for creating, compiling, and managing models with
standard configurations and custom metrics.
"""

from typing import Optional, List, Union
import tensorflow as tf
from ml_pipelines.config import ModelConfig


def create_optimizer(
    config: ModelConfig,
    steps_per_epoch: Optional[int] = None
) -> tf.keras.optimizers.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        config: ModelConfig with optimizer settings
        steps_per_epoch: Steps per epoch for learning rate schedules
        
    Returns:
        Configured optimizer
    """
    optimizer_name = config.optimizer.lower()
    
    # Create learning rate (can be scalar or schedule)
    learning_rate = config.learning_rate
    
    # Create optimizer
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=config.gradient_clip_norm
        )
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            clipnorm=config.gradient_clip_norm,
            momentum=0.9
        )
    elif optimizer_name == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=config.gradient_clip_norm
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_loss(config: ModelConfig) -> Union[str, tf.keras.losses.Loss]:
    """
    Create loss function based on configuration.
    
    Args:
        config: ModelConfig with loss settings
        
    Returns:
        Loss function or loss name
    """
    loss_name = config.loss.lower()
    
    # Map common loss names
    loss_map = {
        "mse": "mse",
        "mae": "mae",
        "huber": "huber",
        "binary_crossentropy": "binary_crossentropy",
        "categorical_crossentropy": "categorical_crossentropy",
        "sparse_categorical_crossentropy": "sparse_categorical_crossentropy",
    }
    
    if loss_name in loss_map:
        return loss_map[loss_name]
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def compile_model(
    model: tf.keras.Model,
    config: ModelConfig,
    metrics: Optional[List] = None,
    steps_per_epoch: Optional[int] = None
) -> tf.keras.Model:
    """
    Compile model with standard configuration.
    
    Args:
        model: Uncompiled Keras model
        config: ModelConfig with compilation settings
        metrics: List of metrics to track (default: [])
        steps_per_epoch: Steps per epoch for LR schedules
        
    Returns:
        Compiled Keras model
    """
    # Create optimizer and loss
    optimizer = create_optimizer(config, steps_per_epoch)
    loss = create_loss(config)
    
    # Use provided metrics or empty list
    if metrics is None:
        metrics = []
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print(f"✓ Model compiled:")
    print(f"  Optimizer: {config.optimizer} (lr={config.learning_rate})")
    print(f"  Loss: {config.loss}")
    print(f"  Metrics: {[str(m) for m in metrics]}")
    
    return model


def get_callbacks(
    config: ModelConfig,
    checkpoint_dir: str = "checkpoints",
    monitor: str = "val_loss",
    mode: str = "min"
) -> List[tf.keras.callbacks.Callback]:
    """
    Get standard training callbacks.
    
    Args:
        config: ModelConfig with training settings
        checkpoint_dir: Directory for model checkpoints
        monitor: Metric to monitor for callbacks
        mode: 'min' or 'max' for monitor metric
        
    Returns:
        List of Keras callbacks
    """
    import os
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.keras")
    
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            mode=mode,
            verbose=1
        ),
        
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            mode=mode,
            verbose=0
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=config.reduce_lr_patience,
            min_lr=config.min_learning_rate,
            mode=mode,
            verbose=1
        ),
    ]
    
    print(f"✓ Standard callbacks configured:")
    print(f"  EarlyStopping (patience={config.early_stopping_patience})")
    print(f"  ModelCheckpoint -> {checkpoint_path}")
    print(f"  ReduceLROnPlateau (patience={config.reduce_lr_patience})")
    
    return callbacks
