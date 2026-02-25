"""
onnx_export.py
--------------
Export a trained PyTorch Forecasting TFT model to ONNX format.

The TFT forward() method expects a dict of tensors, which ONNX does not
support natively.  This module provides:

  1. TFTOnnxWrapper — an nn.Module that accepts individual tensor arguments
     and internally constructs the dict that TFT expects.
  2. export_tft_to_onnx() — traces the wrapper with dummy inputs and writes
     the .onnx file.

The exported graph has fixed encoder_length=20 and prediction_length=1.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TFTOnnxWrapper(nn.Module):
    """Thin wrapper that converts flat tensor args into the dict TFT expects.

    ONNX export requires all inputs/outputs to be tensors (no dicts).  This
    module accepts the tensors individually, packs them into the dict
    required by ``TemporalFusionTransformer.forward()``, calls forward,
    and returns only the ``prediction`` tensor — shape (B, 1, 3) for
    three quantile outputs (P10 / P50 / P90).
    """

    def __init__(self, tft_model):
        super().__init__()
        self.tft = tft_model

    def forward(
        self,
        encoder_cat: torch.Tensor,    # (B, encoder_len, n_cat)
        encoder_cont: torch.Tensor,   # (B, encoder_len, n_cont)
        decoder_cat: torch.Tensor,    # (B, pred_len, n_cat)
        decoder_cont: torch.Tensor,   # (B, pred_len, n_cont)
        encoder_lengths: torch.Tensor,  # (B,)
        decoder_lengths: torch.Tensor,  # (B,)
        target_scale: torch.Tensor,     # (B, 2) — center and scale
    ) -> torch.Tensor:
        x = {
            "encoder_cat": encoder_cat,
            "encoder_cont": encoder_cont,
            "decoder_cat": decoder_cat,
            "decoder_cont": decoder_cont,
            "encoder_lengths": encoder_lengths,
            "decoder_lengths": decoder_lengths,
            "target_scale": target_scale,
        }
        output = self.tft(x)
        return output.prediction  # (B, pred_len, n_quantiles)


def _build_dummy_inputs(
    tft_model,
    encoder_length: int = 20,
    prediction_length: int = 1,
    batch_size: int = 1,
) -> Tuple[torch.Tensor, ...]:
    """Build dummy tensors matching the TFT's expected input shapes."""
    n_encoder_cat = len(tft_model.hparams.x_categoricals)
    n_encoder_cont = len(tft_model.hparams.x_reals)

    encoder_cat = torch.zeros(batch_size, encoder_length, n_encoder_cat, dtype=torch.long)
    encoder_cont = torch.zeros(batch_size, encoder_length, n_encoder_cont)
    decoder_cat = torch.zeros(batch_size, prediction_length, n_encoder_cat, dtype=torch.long)
    decoder_cont = torch.zeros(batch_size, prediction_length, n_encoder_cont)
    encoder_lengths = torch.full((batch_size,), encoder_length, dtype=torch.long)
    decoder_lengths = torch.full((batch_size,), prediction_length, dtype=torch.long)
    target_scale = torch.ones(batch_size, 2)  # (center, scale)

    return (
        encoder_cat,
        encoder_cont,
        decoder_cat,
        decoder_cont,
        encoder_lengths,
        decoder_lengths,
        target_scale,
    )


def export_tft_to_onnx(
    tft_model,
    output_path: str,
    encoder_length: int = 20,
    prediction_length: int = 1,
    opset_version: int = 17,
) -> str:
    """Export a TFT model to ONNX format.

    Parameters
    ----------
    tft_model : TemporalFusionTransformer
        Trained model instance.
    output_path : str
        Directory where model.onnx + metadata will be saved.
    encoder_length : int
        Fixed encoder window size (default 20).
    prediction_length : int
        Fixed prediction horizon (default 1).
    opset_version : int
        ONNX opset (default 17).

    Returns
    -------
    str
        Path to the exported .onnx file.
    """
    os.makedirs(output_path, exist_ok=True)
    onnx_file = os.path.join(output_path, "model.onnx")

    tft_model.eval()
    tft_model.cpu()
    wrapper = TFTOnnxWrapper(tft_model)
    wrapper.eval()

    dummy = _build_dummy_inputs(tft_model, encoder_length, prediction_length)
    input_names = [
        "encoder_cat",
        "encoder_cont",
        "decoder_cat",
        "decoder_cont",
        "encoder_lengths",
        "decoder_lengths",
        "target_scale",
    ]
    output_names = ["prediction"]

    dynamic_axes = {
        "encoder_cat": {0: "batch"},
        "encoder_cont": {0: "batch"},
        "decoder_cat": {0: "batch"},
        "decoder_cont": {0: "batch"},
        "encoder_lengths": {0: "batch"},
        "decoder_lengths": {0: "batch"},
        "target_scale": {0: "batch"},
        "prediction": {0: "batch"},
    }

    logger.info("Exporting TFT to ONNX at %s (opset %d)...", onnx_file, opset_version)

    # PyTorch ≥2.6 defaults to the dynamo-based ONNX exporter, which
    # cannot handle TFT's data-dependent control flow
    # (e.g. ``int(encoder_lengths.max())``).  Force legacy TorchScript
    # exporter when available.
    export_kwargs: Dict = dict(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if torch_version >= (2, 6):
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_file,
        **export_kwargs,
    )

    logger.info("ONNX export complete: %s", onnx_file)

    # Save the feature metadata that the serving layer needs to map
    # JSON request fields → tensor columns in the correct order.
    metadata = {
        "x_categoricals": list(tft_model.hparams.x_categoricals),
        "x_reals": list(tft_model.hparams.x_reals),
        "encoder_length": encoder_length,
        "prediction_length": prediction_length,
        "quantiles": [0.1, 0.5, 0.9],
    }
    meta_path = os.path.join(output_path, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to %s", meta_path)

    return onnx_file
