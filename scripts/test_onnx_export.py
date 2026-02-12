"""Quick test: build TFT → export ONNX → run inference with onnxruntime."""
import torch
import numpy as np
import json
import tempfile
import os

from mlops_pipeline.tests.test_rush_hour_viz import _build_synthetic_data, _build_dataset
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

print("Building synthetic data...")
df = _build_synthetic_data()
ds = _build_dataset(df)

print("Creating small TFT model...")
model = TemporalFusionTransformer.from_dataset(
    ds,
    learning_rate=0.001,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    output_size=3,
)
model.eval()

print(f"  x_categoricals: {model.hparams.x_categoricals}")
print(f"  x_reals: {model.hparams.x_reals}")
print(f"  n_cat={len(model.hparams.x_categoricals)}, n_real={len(model.hparams.x_reals)}")

print("Exporting to ONNX...")
from mlops_pipeline.src.serving.onnx_export import export_tft_to_onnx

out_dir = tempfile.mkdtemp(prefix="onnx_test_")
onnx_path = export_tft_to_onnx(model, out_dir, encoder_length=20, prediction_length=1)
print(f"  Exported to: {onnx_path}")
print(f"  File size: {os.path.getsize(onnx_path) / 1024:.0f} KB")

# Check metadata file
meta_path = os.path.join(out_dir, "model_metadata.json")
with open(meta_path) as f:
    meta = json.load(f)
print(f"  Metadata keys: {list(meta.keys())}")
print(f"  Quantiles: {meta['quantiles']}")

# Verify with onnxruntime
print("Running ONNX inference...")
import onnxruntime as ort

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

from mlops_pipeline.src.serving.onnx_export import _build_dummy_inputs

dummy = _build_dummy_inputs(model, encoder_length=20, prediction_length=1)
feeds = {
    "encoder_cat": dummy[0].numpy(),
    "encoder_cont": dummy[1].numpy(),
    "decoder_cat": dummy[2].numpy(),
    "decoder_cont": dummy[3].numpy(),
    "encoder_lengths": dummy[4].numpy(),
    "decoder_lengths": dummy[5].numpy(),
    "target_scale": dummy[6].numpy(),
}
result = session.run(["prediction"], feeds)
pred = result[0]
print(f"  Output shape: {pred.shape}")
print(f"  Quantiles: P10={pred[0, 0, 0]:.3f}, P50={pred[0, 0, 1]:.3f}, P90={pred[0, 0, 2]:.3f}")

# Batch test: 4 samples
print("Batch inference (4 samples)...")
feeds_batch = {k: np.repeat(v, 4, axis=0) for k, v in feeds.items()}
result_batch = session.run(["prediction"], feeds_batch)
print(f"  Batch output shape: {result_batch[0].shape}")
assert result_batch[0].shape[0] == 4, "Batch dimension mismatch!"

print()
print("=" * 50)
print("ONNX EXPORT + INFERENCE: SUCCESS")
print("=" * 50)
