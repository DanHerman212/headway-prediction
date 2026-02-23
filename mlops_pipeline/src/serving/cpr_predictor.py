"""
cpr_predictor.py
----------------
Vertex AI Custom Prediction Routine (CPR) for headway TFT model.

Implements the google.cloud.aiplatform.prediction.Predictor interface:
  - load():        Download & load model + fitted dataset from GCS artifacts
  - preprocess():  Convert JSON instances → list of DataFrames ready for inference
  - predict():     Run TFT inference via pytorch-forecasting
  - postprocess(): Format quantile outputs into the API response

The CPR SDK handles Flask/gunicorn, health checks, and request routing.
No manual HTTP server code needed.
"""

import logging
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from google.cloud.aiplatform.prediction import Predictor

logger = logging.getLogger(__name__)

# Type alias for a single prediction result
PredictionResult = Dict[str, Any]


class HeadwayPredictor(Predictor):
    """CPR Predictor for the headway TFT model.

    Artifacts expected in the model directory (uploaded by register_model):
      - model_full.pt         — full TFT model (architecture + weights)
      - training_dataset.pt   — fitted TimeSeriesDataSet with encoders/normalizers
    """

    def __init__(self):
        super().__init__()
        self._model = None
        self._training_dataset = None

    # ── load ─────────────────────────────────────────────────────────
    def load(self, artifacts_uri: str) -> None:
        """Load model and fitted dataset from the artifacts directory.

        The CPR SDK may pass either a local path (if it pre-downloaded
        artifacts) or a GCS URI.  We handle both cases.
        """
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

        logger.info("Loading artifacts from %s", artifacts_uri)

        # If the SDK passed a GCS URI, download artifacts locally first
        if artifacts_uri.startswith("gs://"):
            local_dir = self._download_gcs_artifacts(artifacts_uri)
        else:
            local_dir = artifacts_uri

        logger.info("Resolved local artifact dir: %s", local_dir)

        ds_path = self._find_file(local_dir, "training_dataset.pt")
        logger.info("Loading fitted TimeSeriesDataSet from %s", ds_path)
        self._training_dataset = torch.load(ds_path, weights_only=False, map_location="cpu")
        logger.info(
            "  max_encoder_length=%d, max_prediction_length=%d",
            self._training_dataset.max_encoder_length,
            self._training_dataset.max_prediction_length,
        )

        model_path = self._find_file(local_dir, "model_full.pt")
        logger.info("Loading full model from %s", model_path)
        self._model = torch.load(model_path, weights_only=False, map_location="cpu")
        self._model.eval()
        logger.info("Model loaded (type=%s)", type(self._model).__name__)

    # ── preprocess ───────────────────────────────────────────────────
    def preprocess(self, prediction_input: Any) -> List[Dict[str, Any]]:
        """Convert raw JSON request into a list of prepared instance dicts.

        Each instance dict contains:
          - group_id: str
          - dataframe: pd.DataFrame ready for TimeSeriesDataSet.from_dataset()
            (encoder window + 1 fabricated future row)

        Input format (Vertex AI standard):
        {
            "instances": [
                {
                    "group_id": "A_South",
                    "observations": [ {feature_dict}, ... ]  # 10-20 rows
                }
            ]
        }
        """
        instances = prediction_input.get("instances", prediction_input)
        if not isinstance(instances, list):
            instances = [instances]

        prepared = []
        for instance in instances:
            group_id = instance["group_id"]
            observations = instance["observations"]

            # Build DataFrame from observations
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

            # time_idx must exist as integer and be monotonically increasing.
            # We preserve the original physical-time values because the training
            # dataset uses add_relative_time_idx=True, which derives a
            # relative_time_idx feature from the raw time_idx gaps.
            # Re-indexing to contiguous integers changes that derived feature
            # and breaks predictions.
            if "time_idx" not in df.columns:
                df["time_idx"] = range(len(df))
            else:
                df["time_idx"] = df["time_idx"].astype(int)

            # target column must exist (required by TimeSeriesDataSet)
            if "service_headway" not in df.columns:
                df["service_headway"] = 0.0

            # NOTE: We intentionally do NOT fabricate a dummy future row.
            # With predict=True, TimeSeriesDataSet places the LAST
            # observation in the decoder (as the prediction target) and
            # all preceding observations in the encoder — exactly matching
            # how the training DataLoader builds windows.
            # Adding a dummy row shifts the encoder boundary by one step,
            # corrupting the per-window EncoderNormalizer statistics and
            # producing catastrophically wrong predictions.

            prepared.append({"group_id": group_id, "dataframe": df})

        return prepared

    # ── predict ──────────────────────────────────────────────────────
    def predict(self, instances: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Run TFT inference for each prepared instance.

        Uses TimeSeriesDataSet.from_dataset(predict=True) to apply the
        exact same encoders, normalizers, and scalers from training.
        model.predict(mode="quantiles") returns denormalized values in
        the original scale (headway minutes).
        """
        from pytorch_forecasting import TimeSeriesDataSet

        results = []
        for inst in instances:
            group_id = inst["group_id"]
            df = inst["dataframe"]

            try:
                predict_dataset = TimeSeriesDataSet.from_dataset(
                    self._training_dataset,
                    df,
                    predict=True,
                    stop_randomization=True,
                )
            except Exception as e:
                logger.error("Failed to build dataset for %s: %s", group_id, e)
                results.append({
                    "group_id": group_id,
                    "error": str(e),
                    "headway_p10": None,
                    "headway_p50": None,
                    "headway_p90": None,
                })
                continue

            dataloader = predict_dataset.to_dataloader(
                batch_size=1, shuffle=False, num_workers=0,
            )

            with torch.no_grad():
                predictions = self._model.predict(dataloader, mode="quantiles")

            # predictions shape: (n_samples, prediction_length, n_quantiles)
            quantiles = predictions[0, 0, :].detach().cpu().numpy()

            results.append({
                "group_id": group_id,
                "headway_p10": round(float(quantiles[0]), 2),
                "headway_p50": round(float(quantiles[1]), 2),
                "headway_p90": round(float(quantiles[2]), 2),
            })

        return results

    # ── postprocess ──────────────────────────────────────────────────
    def postprocess(self, prediction_results: List[PredictionResult]) -> Dict[str, Any]:
        """Wrap results in the Vertex AI response format."""
        return {"predictions": prediction_results}

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _download_gcs_artifacts(gcs_uri: str) -> str:
        """Download all files from a GCS URI to a local temp directory."""
        from google.cloud import storage as gcs_storage
        import tempfile

        local_dir = tempfile.mkdtemp(prefix="cpr_artifacts_")
        logger.info("Downloading GCS artifacts from %s to %s", gcs_uri, local_dir)

        # Parse gs://bucket/prefix
        path = gcs_uri.replace("gs://", "")
        bucket_name = path.split("/")[0]
        prefix = "/".join(path.split("/")[1:])
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        client = gcs_storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            raise FileNotFoundError(
                f"No artifacts found at {gcs_uri}. "
                f"Bucket={bucket_name}, prefix={prefix}"
            )

        for blob in blobs:
            # Skip "directory" markers
            if blob.name.endswith("/"):
                continue
            rel_path = blob.name[len(prefix):]
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            logger.info("  Downloaded %s (%.1f MB)", rel_path, blob.size / 1e6)

        return local_dir

    @staticmethod
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
