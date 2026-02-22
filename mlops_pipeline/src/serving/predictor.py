"""
predictor.py
------------
Native PyTorch serving for the headway TFT model.

Uses pytorch-forecasting's own TimeSeriesDataSet.from_dataset() for
preprocessing and model.predict() for inference + denormalization.
This guarantees that feature scaling, encoding, and denormalization are
identical to training — no manual reimplementation of transforms.

Runs as a Flask/gunicorn HTTP server inside a Vertex AI Prediction container.

Request format
--------------
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
        "day_of_week": 3,
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

Response format
---------------
{
  "predictions": [{
    "group_id": "A_South",
    "headway_p10": 4.1,
    "headway_p50": 5.3,
    "headway_p90": 7.2
  }]
}
"""

import logging
import os
import time
from typing import Any, Dict

import pandas as pd
import torch
from flask import Flask, jsonify, request
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# ---- Global state (loaded once at startup) ----
_model: TemporalFusionTransformer = None
_training_dataset: TimeSeriesDataSet = None

_LOCAL_MODEL_DIR = "/tmp/model_artifacts"


# ═══════════════════════════════════════════════════════════════════════
# Artifact loading
# ═══════════════════════════════════════════════════════════════════════


def _download_from_gcs(gcs_uri: str, local_dir: str) -> None:
    """Download model artifacts from a GCS URI to a local directory."""
    from google.cloud import storage as gcs_storage

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
        rel_path = blob.name[len(prefix):] if prefix else blob.name
        if not rel_path or rel_path.endswith("/"):
            continue
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info("  Downloaded %s (%.1f KB)", rel_path,
                     os.path.getsize(local_path) / 1024)


def _find_file(base_dir: str, filename: str) -> str:
    """Find a file in base_dir or any subdirectory."""
    candidate = os.path.join(base_dir, filename)
    if os.path.exists(candidate):
        return candidate
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(
        f"{filename} not found in {base_dir}. "
        f"Contents: {os.listdir(base_dir) if os.path.isdir(base_dir) else 'N/A'}"
    )


def _load_model():
    """Load the PyTorch model + fitted TimeSeriesDataSet from artifacts.

    Expected artifacts (uploaded by register_model):
        - model_state_dict.pt   — trained model weights
        - training_dataset.pt   — fitted TimeSeriesDataSet with all
                                  encoders, normalizers, and scalers
    """
    global _model, _training_dataset

    storage_uri = os.environ.get("AIP_STORAGE_URI", "/mnt/models")

    if storage_uri.startswith("gs://"):
        _download_from_gcs(storage_uri, _LOCAL_MODEL_DIR)
        model_dir = _LOCAL_MODEL_DIR
    else:
        model_dir = storage_uri

    # 1. Load the fitted dataset (carries all preprocessing logic)
    ds_path = _find_file(model_dir, "training_dataset.pt")
    logger.info("Loading fitted TimeSeriesDataSet from %s", ds_path)
    _training_dataset = torch.load(ds_path, weights_only=False, map_location="cpu")
    logger.info("  max_encoder_length=%d, max_prediction_length=%d",
                _training_dataset.max_encoder_length,
                _training_dataset.max_prediction_length)

    # 2. Load the full model (architecture + weights + hparams)
    model_path = _find_file(model_dir, "model_full.pt")
    logger.info("Loading full model from %s", model_path)
    _model = torch.load(model_path, weights_only=False, map_location="cpu")
    _model.eval()
    logger.info("Model loaded and set to eval mode (type=%s)",
                 type(_model).__name__)


# ═══════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════


def _build_dataframe(instance: Dict[str, Any]) -> pd.DataFrame:
    """Convert a request instance to a DataFrame matching the training schema.

    The caller sends raw (unscaled) feature values — the same format as the
    training parquet.  TimeSeriesDataSet.from_dataset() will handle all
    encoding, scaling, and normalization identically to training.
    """
    group_id = instance["group_id"]
    observations = instance["observations"]

    rows = []
    for obs in observations:
        row = dict(obs)
        row["group_id"] = group_id
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure categoricals are strings (same as clean_dataset)
    for col in ["group_id", "route_id", "regime_id", "track_id", "preceding_route_id"]:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    # time_idx must be present and monotonically increasing per group
    if "time_idx" not in df.columns:
        df["time_idx"] = range(len(df))

    # target must exist (it's the label column)
    if "service_headway" not in df.columns:
        df["service_headway"] = 0.0

    return df


def _predict_instance(instance: Dict[str, Any]) -> Dict[str, Any]:
    """Run prediction for a single instance.

    1. Build a DataFrame from the raw observations
    2. Create a TimeSeriesDataSet via from_dataset() — applies ALL the
       same encoders, normalizers, and scalers as training
    3. Call model.predict() — runs forward() + transform_output() which
       returns denormalized values in the original scale (minutes)
    4. Extract P10 / P50 / P90 from the quantile output
    """
    group_id = instance["group_id"]
    df = _build_dataframe(instance)

    # Add a "future" row so the dataset can form a decoder window.
    # from_dataset(predict=True) needs at least 1 step beyond encoder.
    last_row = df.iloc[-1].copy()
    last_row["time_idx"] = int(last_row["time_idx"]) + 1
    last_row["service_headway"] = 0.0  # target is unknown
    df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)

    # Build dataset using from_dataset — inherits fitted encoders/normalizers
    try:
        predict_dataset = TimeSeriesDataSet.from_dataset(
            _training_dataset,
            df,
            predict=True,
            stop_randomization=True,
        )
    except Exception as e:
        logger.error("Failed to build prediction dataset for %s: %s", group_id, e)
        return {
            "group_id": group_id,
            "error": str(e),
            "headway_p10": None,
            "headway_p50": None,
            "headway_p90": None,
        }

    dataloader = predict_dataset.to_dataloader(
        batch_size=1, shuffle=False, num_workers=0,
    )

    # Get raw model output + batch metadata for manual denormalization.
    # The model's internal predictions are in NORMALIZED space (near 0)
    # due to GroupNormalizer(softplus). We must explicitly denormalize
    # via transform_output() to convert back to headway minutes.
    with torch.no_grad():
        raw_output = _model.predict(
            dataloader,
            mode="raw",
            return_x=True,
        )

    # raw_output.output["prediction"]: (n_samples, pred_len, n_quantiles) — normalized
    # raw_output.x["target_scale"]: (n_samples, 2) — (center, scale) per sample
    raw_prediction = raw_output.output["prediction"]
    target_scale = raw_output.x["target_scale"]

    # Denormalize: reverses GroupNormalizer(softplus) → real minutes
    denormalized = _model.transform_output(
        raw_prediction, target_scale=target_scale,
    )

    # denormalized shape: (n_samples, prediction_length, n_quantiles)
    quantiles = denormalized[0, 0, :].detach().cpu().numpy()

    return {
        "group_id": group_id,
        "headway_p10": round(float(quantiles[0]), 2),
        "headway_p50": round(float(quantiles[1]), 2),
        "headway_p90": round(float(quantiles[2]), 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# HTTP endpoints
# ═══════════════════════════════════════════════════════════════════════


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests (Vertex AI format)."""
    start = time.time()
    body = request.get_json(force=True)
    instances = body.get("instances", [])

    predictions = []
    for instance in instances:
        result = _predict_instance(instance)
        predictions.append(result)

    elapsed_ms = (time.time() - start) * 1000
    logger.info("Processed %d instances in %.1f ms", len(instances), elapsed_ms)
    return jsonify({"predictions": predictions})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Vertex AI."""
    if _model is None:
        return jsonify({"status": "unhealthy", "reason": "model not loaded"}), 503
    return jsonify({"status": "healthy"})


# ═══════════════════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════════════════


def _safe_load_model():
    """Attempt to load model, logging errors instead of crashing."""
    try:
        _load_model()
        logger.info("Model loaded successfully — ready to serve")
    except Exception:
        logger.exception("Failed to load model at startup — /health will return 503")


if __name__ == "__main__":
    _load_model()
    port = int(os.environ.get("AIP_HTTP_PORT", "8080"))
    logger.info("Starting prediction server on port %d", port)
    app.run(host="0.0.0.0", port=port)
else:
    # Loaded by gunicorn --preload
    _safe_load_model()
