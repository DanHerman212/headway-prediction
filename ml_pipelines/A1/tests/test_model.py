"""Unit tests for model architecture and training."""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import get_model


def test_model_architecture(mock_config):
    """Test that model has correct architecture and outputs."""
    model = get_model(compile=False)
    
    # Check model has two outputs
    assert len(model.outputs) == 2
    # Output names may vary, just check we have the layers
    layer_names = [layer.name for layer in model.layers]
    assert any('route' in name for name in layer_names), f"No route layer found in {layer_names}"
    assert any('headway' in name for name in layer_names), f"No headway layer found in {layer_names}"
    
    # Check output shapes
    dummy_input = tf.random.normal((1, mock_config.LOOKBACK_WINDOW, 8))
    outputs = model(dummy_input)
    
    if isinstance(outputs, dict):
        route_out = outputs['route_output']
        headway_out = outputs['headway_output']
    else:
        route_out, headway_out = outputs
    
    assert route_out.shape == (1, 3), f"Route output shape mismatch: {route_out.shape}"
    assert headway_out.shape == (1, 1), f"Headway output shape mismatch: {headway_out.shape}"


def test_model_compilation(mock_config):
    """Test that model compiles without errors."""
    model = get_model(compile=True)
    
    # Should already be compiled by get_model
    assert model.optimizer is not None
    assert model.compiled_loss is not None


def test_model_forward_pass(mock_config):
    """Test that model can perform forward pass without errors."""
    model = get_model(compile=False)
    
    # Create batch of sample data
    batch_size = 8
    X_batch = tf.random.normal((batch_size, mock_config.LOOKBACK_WINDOW, 8))
    
    # Forward pass
    outputs = model(X_batch, training=False)
    
    # Verify outputs
    if isinstance(outputs, dict):
        route_out = outputs['route_output']
        headway_out = outputs['headway_output']
    else:
        route_out, headway_out = outputs
    
    assert route_out.shape[0] == batch_size
    assert headway_out.shape[0] == batch_size
    
    # Route output should be probabilities (softmax)
    route_sums = tf.reduce_sum(route_out, axis=1)
    tf.debugging.assert_near(route_sums, tf.ones(batch_size), atol=1e-5)


def test_model_training_step(mock_config, sample_preprocessed_data):
    """Test that model can perform a single training step."""
    model = get_model(compile=True)
    
    # Create properly shaped batch for lookback window
    batch_size = 8
    lookback = mock_config.LOOKBACK_WINDOW
    
    # Create sequences with proper shape (batch, lookback, features)
    X_batch = tf.constant(
        sample_preprocessed_data[:lookback + batch_size].reshape(1, -1, 8)[:, :lookback, :],
        dtype=tf.float32
    )
    # Repeat to make full batch
    X_batch = tf.repeat(X_batch, batch_size, axis=0)
    
    y_route = tf.constant([[1, 0, 0]] * batch_size, dtype=tf.float32)
    y_headway = tf.constant([[5.0]] * batch_size, dtype=tf.float32)
    
    # Training step
    with tf.GradientTape() as tape:
        outputs = model(X_batch, training=True)
        if isinstance(outputs, dict):
            route_out = outputs['route_output']
            headway_out = outputs['headway_output']
        else:
            route_out, headway_out = outputs
        
        # Compute losses
        route_loss = tf.keras.losses.categorical_crossentropy(y_route, route_out)
        headway_loss = tf.keras.losses.huber(y_headway, headway_out)
        total_loss = tf.reduce_mean(route_loss) + tf.reduce_mean(headway_loss)
    
    # Check that loss is finite
    assert tf.math.is_finite(total_loss)
    
    # Check that gradients can be computed
    gradients = tape.gradient(total_loss, model.trainable_variables)
    assert all(g is not None for g in gradients)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
