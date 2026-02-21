"""
predictor.py
------------
Custom Prediction Routine (CPR) for the headway TFT model.

Runs as an HTTP server inside a Vertex AI Prediction container.
Loads the ONNX model + metadata at startup, then handles prediction
requests for individual group_ids.

Request format:
{
  "instances": [{
    "group_id": "A_South",
    "observations": [
      {
        "service_headway": 5.2,
        "preceding_train_gap": 4.8,
        "upstream_headway_14th": 5.5,
        "travel_time_14th": 2.1,
        "travel_time_14th_deviation": 0.05,
        "travel_time_23rd": 1.8,
        "travel_time_23rd_deviation": 0.03,
        "travel_time_34th": 2.3,
        "travel_time_34th_deviation": 0.04,
        "stops_at_23rd": 1.0,
        "hour_sin": 0.87,
        "hour_cos": -0.5,
        "time_idx": 14200,
        "empirical_median": 5.0,
        "route_id": "A",
        "regime_id": "AM_RUSH",
        "track_id": "local",
        "preceding_route_id": "A"
      }
      // ... 20 total observations (encoder window)
    ]
  }]
}

Response format:
{
  "predictions": [{
    "group_id": "A_South",
    "headway_p10": 4.1,
    "headway_p50": 5.3,
    "headway_p90": 7.2
  }]
}
"""

import json
import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# ---- Global state (loaded once at startup) ----
_session: ort.InferenceSession = None
_metadata: Dict[str, Any] = None
_dataset_params: Dict[str, Any] = None
_scaler_lookup: Dict[str, tuple] = {}  # col_name -> (center, scale)


# Required artifact files the model needs to serve
_REQUIRED_ARTIFACTS = ["model.onnx", "model_metadata.json", "dataset_params.json"]
_LOCAL_MODEL_DIR = "/tmp/model_artifacts"


def _download_from_gcs(gcs_uri: str, local_dir: str) -> None:
    """Download model artifacts from a GCS URI to a local directory.

    Vertex AI sets AIP_STORAGE_URI to a gs:// URI for custom containers.
    The container must download the artifacts itself.
    """
    from google.cloud import storage as gcs_storage

    # Parse gs://bucket/prefix
    path = gcs_uri.replace("gs://", "")
    bucket_name = path.split("/")[0]
    prefix = "/".join(path.split("/")[1:])
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    logger.info("Downloading artifacts from gs://%s/%s to %s",
                bucket_name, prefix, local_dir)
    os.makedirs(local_dir, exist_ok=True)

    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No blobs found at {gcs_uri}")

    for blob in blobs:
        # Compute relative path from prefix
        rel_path = blob.name[len(prefix):] if prefix else blob.name
        if not rel_path or rel_path.endswith("/"):
            continue
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info("  Downloaded %s (%.1f KB)", rel_path,
                    os.path.getsize(local_path) / 1024)


def _load_model():
    """Load ONNX model + metadata from the model directory.

    Vertex AI sets AIP_STORAGE_URI to a gs:// URI for custom containers.
    This function downloads from GCS first, then loads locally.
    """
    global _session, _metadata, _dataset_params

    storage_uri = os.environ.get("AIP_STORAGE_URI", "/mnt/models")

    # If it's a GCS path, download artifacts locally first
    if storage_uri.startswith("gs://"):
        _download_from_gcs(storage_uri, _LOCAL_MODEL_DIR)
        model_dir = _LOCAL_MODEL_DIR
    else:
        model_dir = storage_uri

    # In some deployments the artifacts are nested; handle both layouts
    onnx_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(onnx_path):
        # Try one level deeper
        for root, _, files in os.walk(model_dir):
            if "model.onnx" in files:
                onnx_path = os.path.join(root, "model.onnx")
                model_dir = root
                break

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"model.onnx not found in {storage_uri} (searched {model_dir}). "
            f"Available files: {os.listdir(model_dir) if os.path.isdir(model_dir) else 'dir not found'}"
        )

    logger.info("Loading ONNX model from %s", onnx_path)
    _session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path) as f:
        _metadata = json.load(f)
    logger.info(
        "Loaded metadata: %d categoricals, %d reals",
        len(_metadata["x_categoricals"]),
        len(_metadata["x_reals"]),
    )

    params_path = os.path.join(model_dir, "dataset_params.json")
    with open(params_path) as f:
        _dataset_params = json.load(f)
    logger.info("Loaded dataset params with %d encoder mappings",
                len(_dataset_params.get("categorical_encoders", {})))

    # Build per-feature scaler lookup: col_name -> (center, scale)
    global _scaler_lookup
    _scaler_lookup = {}
    for col, sp in _dataset_params.get("scaler_params", {}).items():
        _scaler_lookup[col] = (float(sp["center"]), float(sp["scale"]))
    logger.info("Loaded %d feature scalers: %s",
                len(_scaler_lookup), list(_scaler_lookup.keys()))


def _encode_categorical(col_name: str, value: str) -> int:
    """Map a categorical string value to its integer encoding."""
    encoders = _dataset_params.get("categorical_encoders", {})
    if col_name in encoders:
        mapping = encoders[col_name]
        return mapping.get(str(value), 0)  # default to 0 for unknown
    return 0


def _get_target_scale(group_id: str) -> np.ndarray:
    """Get (center, scale) for the group's normalizer."""
    norm_params = _dataset_params.get("normalizer_params", {})
    center_map = norm_params.get("center", {})
    scale_map = norm_params.get("scale", {})
    if str(group_id) not in center_map or str(group_id) not in scale_map:
        logger.warning(
            "group_id '%s' not found in normalizer_params "
            "(available: %s) — using fallback center=0.0, scale=1.0. "
            "Predictions will NOT be denormalized!",
            group_id,
            list(center_map.keys())[:5],
        )
    center = float(center_map.get(str(group_id), 0.0))
    scale = float(scale_map.get(str(group_id), 1.0))
    return np.array([[center, scale]], dtype=np.float32)


def _normalize_target(value: float, group_id: str) -> float:
    """Apply softplus then group z-score to service_headway.

    Matches the GroupNormalizer(transformation='softplus') used during training.
    """
    sp = float(np.log1p(np.exp(value)))  # softplus transform
    norm_params = _dataset_params.get("normalizer_params", {})
    center = float(norm_params.get("center", {}).get(str(group_id), 0.0))
    scale = float(norm_params.get("scale", {}).get(str(group_id), 1.0))
    return (sp - center) / scale


def _scale_feature(col: str, value: float) -> float:
    """Apply z-score scaling if this column has fitted scaler params."""
    if col in _scaler_lookup:
        center, scale = _scaler_lookup[col]
        if scale == 0.0:
            return 0.0
        return (value - center) / scale
    return value


def _preprocess(instance: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert a single request instance to ONNX input tensors.

    Each instance has a group_id and a list of 20 observations forming
    the encoder window.  The decoder window is a single step that we
    populate with the "known future" features from the last observation
    (time features, regime, etc.).
    """
    observations = instance["observations"]
    encoder_length = _metadata["encoder_length"]
    prediction_length = _metadata["prediction_length"]
    cat_cols = _metadata["x_categoricals"]
    real_cols = _metadata["x_reals"]

    # Pad or truncate to exact encoder length
    if len(observations) > encoder_length:
        observations = observations[-encoder_length:]
    elif len(observations) < encoder_length:
        # Pad from the left with copies of the first observation
        pad_count = encoder_length - len(observations)
        observations = [observations[0]] * pad_count + observations

    group_id = instance["group_id"]

    # Build encoder tensors
    encoder_cat = np.zeros((1, encoder_length, len(cat_cols)), dtype=np.int64)
    encoder_cont = np.zeros((1, encoder_length, len(real_cols)), dtype=np.float32)

    # Pre-fetch target scale for the group — also used to fill the
    # service_headway_center / service_headway_scale synthetic features
    # that add_target_scales=True injects into x_reals.
    ts = _get_target_scale(group_id)  # shape (1, 2)
    t_center, t_scale = float(ts[0, 0]), float(ts[0, 1])

    for t, obs in enumerate(observations):
        for ci, col in enumerate(cat_cols):
            encoder_cat[0, t, ci] = _encode_categorical(col, obs.get(col, ""))
        for ri, col in enumerate(real_cols):
            if col == "relative_time_idx":
                # Positional offset: -19, -18, ..., -1, 0
                encoder_cont[0, t, ri] = float(t - encoder_length + 1)
            elif col == "encoder_length":
                encoder_cont[0, t, ri] = float(encoder_length)
            elif col == "service_headway_center":
                encoder_cont[0, t, ri] = t_center
            elif col == "service_headway_scale":
                encoder_cont[0, t, ri] = t_scale
            elif col == "service_headway":
                raw = float(obs.get(col, 0.0))
                encoder_cont[0, t, ri] = _normalize_target(raw, group_id)
            else:
                raw = float(obs.get(col, 0.0))
                encoder_cont[0, t, ri] = _scale_feature(col, raw)

    # Build decoder tensors (1 step — use last observation's known features)
    last_obs = observations[-1]
    decoder_cat = np.zeros((1, prediction_length, len(cat_cols)), dtype=np.int64)
    decoder_cont = np.zeros((1, prediction_length, len(real_cols)), dtype=np.float32)

    for ci, col in enumerate(cat_cols):
        decoder_cat[0, 0, ci] = _encode_categorical(col, last_obs.get(col, ""))
    for ri, col in enumerate(real_cols):
        if col == "relative_time_idx":
            decoder_cont[0, 0, ri] = 1.0  # one step ahead of encoder
        elif col == "encoder_length":
            decoder_cont[0, 0, ri] = float(encoder_length)
        elif col == "service_headway_center":
            decoder_cont[0, 0, ri] = t_center
        elif col == "service_headway_scale":
            decoder_cont[0, 0, ri] = t_scale
        elif col == "time_idx":
            raw = float(last_obs.get(col, 0)) + 1
            decoder_cont[0, 0, ri] = _scale_feature(col, raw)
        elif col == "service_headway":
            decoder_cont[0, 0, ri] = 0.0  # target is unknown in decoder
        else:
            raw = float(last_obs.get(col, 0.0))
            decoder_cont[0, 0, ri] = _scale_feature(col, raw)

    return {
        "encoder_cat": encoder_cat,
        "encoder_cont": encoder_cont,
        "decoder_cat": decoder_cat,
        "decoder_cont": decoder_cont,
        "encoder_lengths": np.array([encoder_length], dtype=np.int64),
        "decoder_lengths": np.array([prediction_length], dtype=np.int64),
        "target_scale": ts,
    }


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests.

    Accepts Vertex AI's standard {'instances': [...]} format.
    """
    start = time.time()
    body = request.get_json(force=True)
    instances = body.get("instances", [])

    predictions = []
    for instance in instances:
        inputs = _preprocess(instance)

        # Run ONNX inference
        result = _session.run(
            ["prediction"],
            inputs,
        )
        # result[0] shape: (1, 1, 3) — batch=1, pred_len=1, quantiles=3
        quantiles = result[0][0, 0, :]  # (3,) — in normalized z-score space

        # Denormalize: ONNX wraps TFT.forward() which returns normalized
        # predictions.  The GroupNormalizer(transformation='softplus')
        # inverse is:
        #   1) undo z-score:  x = q * scale + center
        #   2) softplus:      y = log(1 + exp(x))   [reverses the
        #      softplus_inv that was applied to raw headways during training]
        center, scale = inputs["target_scale"][0]  # (center, scale)
        z_undone = quantiles * scale + center
        # Numerically stable softplus (identity for large x to avoid overflow)
        minutes = np.where(z_undone > 20.0, z_undone, np.log1p(np.exp(z_undone)))

        predictions.append({
            "group_id": instance["group_id"],
            "headway_p10": round(float(minutes[0]), 2),
            "headway_p50": round(float(minutes[1]), 2),
            "headway_p90": round(float(minutes[2]), 2),
        })

    elapsed_ms = (time.time() - start) * 1000
    logger.info("Processed %d instances in %.1f ms", len(instances), elapsed_ms)

    return jsonify({"predictions": predictions})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Vertex AI."""
    if _session is None:
        return jsonify({"status": "unhealthy", "reason": "model not loaded"}), 503
    return jsonify({"status": "healthy"})


def _safe_load_model():
    """Attempt to load model, logging errors instead of crashing."""
    try:
        _load_model()
        logger.info("Model loaded successfully")
    except Exception:
        logger.exception("Failed to load model at startup — /health will return 503")


if __name__ == "__main__":
    _load_model()  # Fail fast in direct execution
    port = int(os.environ.get("AIP_HTTP_PORT", "8080"))
    logger.info("Starting prediction server on port %d", port)
    app.run(host="0.0.0.0", port=port)
else:
    # When imported by gunicorn --preload, load model but don't crash the
    # process — let the health check report 503 so logs are visible.
    _safe_load_model()
