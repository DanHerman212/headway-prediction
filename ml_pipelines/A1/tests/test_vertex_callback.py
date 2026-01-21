"""
Test VertexAIMetricsCallback

Ensures the custom callback properly logs metrics to Vertex AI Experiments during training.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import VertexAIMetricsCallback


def test_vertex_callback_logs_metrics():
    """Test that the callback logs metrics to Vertex AI run object."""
    # Mock Vertex AI run object
    mock_vertex_run = Mock()
    mock_vertex_run.log_metrics = MagicMock()
    
    # Create callback
    callback = VertexAIMetricsCallback(vertex_run=mock_vertex_run)
    
    # Simulate epoch end with metrics
    test_logs = {
        'loss': 0.5,
        'val_loss': 0.6,
        'route_output_accuracy': 0.85,
        'val_route_output_accuracy': 0.82,
        'headway_output_mae_seconds': 45.2,
        'val_headway_output_mae_seconds': 48.1
    }
    
    # Call on_epoch_end
    callback.on_epoch_end(epoch=0, logs=test_logs)
    
    # Verify log_metrics was called
    assert mock_vertex_run.log_metrics.called, "log_metrics should be called"
    
    # Get the logged metrics
    call_args = mock_vertex_run.log_metrics.call_args
    logged_metrics = call_args[0][0]
    
    # Verify all metrics were logged
    assert len(logged_metrics) == len(test_logs), "All metrics should be logged"
    for key in test_logs.keys():
        assert key in logged_metrics, f"{key} should be in logged metrics"
        assert isinstance(logged_metrics[key], float), f"{key} should be converted to float"
    
    print("✓ Callback logs all metrics correctly")


def test_vertex_callback_handles_none_run():
    """Test that the callback handles None vertex_run gracefully."""
    # Create callback with None vertex_run
    callback = VertexAIMetricsCallback(vertex_run=None)
    
    # Simulate epoch end (should not raise error)
    test_logs = {'loss': 0.5}
    callback.on_epoch_end(epoch=0, logs=test_logs)
    
    print("✓ Callback handles None vertex_run gracefully")


def test_vertex_callback_handles_none_logs():
    """Test that the callback handles None logs gracefully."""
    mock_vertex_run = Mock()
    mock_vertex_run.log_metrics = MagicMock()
    
    callback = VertexAIMetricsCallback(vertex_run=mock_vertex_run)
    
    # Simulate epoch end with None logs (should not raise error)
    callback.on_epoch_end(epoch=0, logs=None)
    
    # Verify log_metrics was NOT called
    assert not mock_vertex_run.log_metrics.called, "log_metrics should not be called with None logs"
    
    print("✓ Callback handles None logs gracefully")


def test_vertex_callback_handles_exception():
    """Test that the callback handles exceptions gracefully."""
    # Mock Vertex AI run object that raises exception
    mock_vertex_run = Mock()
    mock_vertex_run.log_metrics = MagicMock(side_effect=Exception("Test exception"))
    
    callback = VertexAIMetricsCallback(vertex_run=mock_vertex_run)
    
    # Simulate epoch end (should not raise error, just log warning)
    test_logs = {'loss': 0.5}
    callback.on_epoch_end(epoch=0, logs=test_logs)
    
    print("✓ Callback handles exceptions gracefully")


def test_callback_integrates_with_create_callbacks():
    """Test that the callback integrates properly with create_callbacks function."""
    from src.train import create_callbacks
    
    # Mock vertex_run
    mock_vertex_run = Mock()
    
    # Create callbacks with vertex_run
    callbacks = create_callbacks(run_name="test_run", vertex_run=mock_vertex_run)
    
    # Check that VertexAIMetricsCallback is in the list
    has_vertex_callback = any(isinstance(cb, VertexAIMetricsCallback) for cb in callbacks)
    assert has_vertex_callback, "VertexAIMetricsCallback should be in callbacks list"
    
    print("✓ Callback integrates with create_callbacks correctly")


if __name__ == "__main__":
    print("="*80)
    print("Testing VertexAIMetricsCallback")
    print("="*80)
    
    try:
        test_vertex_callback_logs_metrics()
        test_vertex_callback_handles_none_run()
        test_vertex_callback_handles_none_logs()
        test_vertex_callback_handles_exception()
        test_callback_integrates_with_create_callbacks()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
