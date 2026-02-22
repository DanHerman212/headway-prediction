"""
deploy_model.py
---------------
Pipeline step that:
  1. Saves the trained model state_dict
  2. Saves the fitted TimeSeriesDataSet (carries all encoders, normalizers,
     and scalers — the serving container uses it via from_dataset() so
     preprocessing is identical to training by construction)
  3. Uploads all artifacts to GCS
  4. Registers the model in Vertex AI Model Registry
"""

import json
import logging
import os
import tempfile
import shutil
from typing import Dict, Any, Optional

import torch
from google.cloud import aiplatform, storage as gcs_storage
from omegaconf import DictConfig
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

logger = logging.getLogger(__name__)

# ---- Constants ----
SERVING_CONTAINER_URI = (
    "us-east1-docker.pkg.dev/realtime-headway-prediction/"
    "mlops-images/headway-serving:latest"
)
MODEL_DISPLAY_NAME = "headway-tft"


def _upload_dir_to_gcs(local_dir: str, gcs_uri: str) -> None:
    """Upload all files in local_dir to the given gs:// prefix."""
    bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])
    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_path = f"{prefix}/{rel_path}"
            bucket.blob(blob_path).upload_from_filename(local_path)
    logger.info("Uploaded %s to %s", local_dir, gcs_uri)


def register_model(
    model: TemporalFusionTransformer,
    training_dataset: TimeSeriesDataSet,
    config: DictConfig,
    test_mae: float,
    test_smape: float,
    run_id: str,
) -> str:
    """Export model to ONNX, upload artifacts to GCS, and register in Vertex AI Model Registry.

    Parameters
    ----------
    model : TemporalFusionTransformer
        Trained model from the training step.
    training_dataset : TimeSeriesDataSet
        Training dataset — needed for encoder/normalizer params.
    config : DictConfig
        Full Hydra config.
    test_mae : float
        Test MAE from evaluation step (logged as model metadata).
    test_smape : float
        Test sMAPE from evaluation step (logged as model metadata).
    run_id : str
        Pipeline run identifier for artifact versioning.

    Returns
    -------
    str
        The Vertex AI Model Registry resource name.
    """
    project = config.infra.project_id
    location = config.infra.location
    artifact_bucket = config.infra.artifact_bucket

    # Determine run ID for artifact versioning
    gcs_model_uri = f"{artifact_bucket}/models/{run_id}"
    logger.info("Deploying model for run: %s", run_id)

    # ---- 1. Save model artifacts to temp dir ----
    local_dir = tempfile.mkdtemp(prefix="deploy_model_")
    try:
        # Full model — includes architecture + weights + hparams
        # This avoids architecture-mismatch issues when loading with
        # only a state_dict (which requires knowing hidden_size, etc.)
        #
        # IMPORTANT: Revert to the base TemporalFusionTransformer class
        # before pickling. The training pipeline uses TFTDisablePlotting
        # (a subclass that no-ops the matplotlib logging hooks to avoid
        # crashes on A100 bf16), but that class lives in the
        # mlops_pipeline package which is NOT installed in the serving
        # container.  The serving container only has pytorch-forecasting,
        # so pickle must reference the base class for torch.load() to
        # succeed.  TFTDisablePlotting adds no state — only method
        # overrides — so this is a safe cast.
        model.__class__ = TemporalFusionTransformer
        model_path = os.path.join(local_dir, "model_full.pt")
        torch.save(model, model_path)
        logger.info("Saved full model to %s (%.1f MB)",
                     model_path, os.path.getsize(model_path) / 1e6)

        # Fitted TimeSeriesDataSet — carries ALL encoders, normalizers,
        # and scalers.  The serving container loads this and uses
        # TimeSeriesDataSet.from_dataset() so preprocessing is identical
        # to training by construction.
        ds_path = os.path.join(local_dir, "training_dataset.pt")
        torch.save(training_dataset, ds_path)
        logger.info("Saved fitted TimeSeriesDataSet to %s (%.1f MB)",
                     ds_path, os.path.getsize(ds_path) / 1e6)

        # ---- 2. Upload to GCS ----
        logger.info("Uploading artifacts to %s", gcs_model_uri)
        _upload_dir_to_gcs(local_dir, gcs_model_uri)

        # ---- 3. Register model in Vertex AI Model Registry ----
        logger.info("Registering model in Vertex AI Model Registry...")
        aiplatform.init(project=project, location=location)

        version_labels = {
            "test_mae": f"{test_mae:.4f}",
            "test_smape": f"{test_smape:.4f}",
            "run_id": run_id,
            "format": "pytorch",
        }

        vertex_model = aiplatform.Model.upload(
            display_name=MODEL_DISPLAY_NAME,
            artifact_uri=gcs_model_uri,
            serving_container_image_uri=SERVING_CONTAINER_URI,
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_ports=[8080],
            labels={k: v.replace(".", "-") for k, v in version_labels.items()},
            description=f"TFT headway model — MAE={test_mae:.4f}, sMAPE={test_smape:.4f}",
        )
        logger.info("Registered model: %s", vertex_model.resource_name)

        return vertex_model.resource_name

    finally:
        shutil.rmtree(local_dir, ignore_errors=True)
