"""
TensorBoard Tracking Module

A lightweight, reusable module for comprehensive TensorBoard logging
in TensorFlow/Keras projects on GCP.

Usage:
    from src.tracking import Tracker, TrackerConfig
    
    tracker = Tracker(TrackerConfig(
        experiment_name="my-experiment",
        run_name="run-001",
        log_dir="gs://my-bucket/tensorboard/run-001",
        histograms=True,
        hparams_dict={"lr": 0.001}
    ))
    
    # With Keras
    model.fit(x, y, callbacks=tracker.keras_callbacks())
    
    # Manual logging
    tracker.log_scalar("custom_metric", value, step=epoch)
    
    tracker.close()
"""

from .config import TrackerConfig
from .tracker import Tracker
from .callbacks import (
    ScalarCallback,
    HistogramCallback,
    HParamsCallback,
    ProfilerCallback,
    GraphCallback,
)

__all__ = [
    "Tracker",
    "TrackerConfig",
    "ScalarCallback",
    "HistogramCallback",
    "HParamsCallback",
    "ProfilerCallback",
    "GraphCallback",
]

__version__ = "0.1.0"
