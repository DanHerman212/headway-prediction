"""
Project-specific visualizations for TensorBoard.

This module contains visualization callbacks that are specific to the
headway prediction project. These are NOT part of the reusable tracking
module - they are tailored to the spatiotemporal nature of headway data.

Usage:
    from src.visualizations import SpatiotemporalCallback
    from src.tracking import Tracker
    
    tracker = Tracker(config)
    viz_callback = SpatiotemporalCallback(
        tracker=tracker,
        validation_data=val_ds,
        freq=5,
    )
    
    model.fit(x, y, callbacks=tracker.keras_callbacks() + [viz_callback])
"""

from .spatiotemporal import SpatiotemporalCallback

__all__ = ["SpatiotemporalCallback"]
