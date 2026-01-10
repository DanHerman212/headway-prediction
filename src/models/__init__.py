"""
Model architectures for headway prediction.

Available:
    - ConvLSTM: Class-based ConvLSTM Encoder-Decoder (preferred)
    - build_convlstm_model: Function wrapper (backwards compatibility)
    - BroadcastScheduleLayer: Custom layer for terminal-to-station broadcasting
"""

from src.models.convlstm import (
    ConvLSTM,
    BroadcastScheduleLayer,
    build_convlstm_model,
)

__all__ = [
    "ConvLSTM",
    "BroadcastScheduleLayer",
    "build_convlstm_model",
]
