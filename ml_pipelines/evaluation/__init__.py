"""Model evaluation and metrics."""

from .metrics import MAESeconds
# Lazy import or direct import recommended to avoid coupling
# from .evaluate_model import ModelEvaluator

__all__ = ["MAESeconds"]
