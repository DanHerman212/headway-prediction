"""
Model architectures for headway prediction.

Available:
    - ConvLSTM: Class-based ConvLSTM Encoder-Decoder (preferred)
    - HeadwayConvLSTM: Alias for ConvLSTM (backwards compatibility)
    - build_convlstm_model: Function wrapper (backwards compatibility)
    - BroadcastScheduleLayer: Custom layer for terminal-to-station broadcasting
    - BroadcastTemporalLayer: Custom layer for temporal feature broadcasting
"""

from src.models.convlstm import (
    ConvLSTM,
    BroadcastScheduleLayer,
    BroadcastTemporalLayer,
    build_convlstm_model,
)

# Backwards compatibility alias
HeadwayConvLSTM = ConvLSTM

__all__ = [
    "ConvLSTM",
    "HeadwayConvLSTM",
    "BroadcastScheduleLayer",
    "BroadcastTemporalLayer",
    "build_convlstm_model",
]
