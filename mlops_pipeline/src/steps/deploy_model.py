"""
deploy_model.py
---------------
ZenML pipeline step that:
  1. Exports the trained TFT to ONNX format
  2. Saves dataset parameters (encoders, normalizer mappings) as JSON
  3. Uploads all artifacts to GCS
  4. Registers the model in Vertex AI Model Registry
  5. Deploys to a Vertex AI Prediction Endpoint
"""

import json
import logging
import os
import tempfile
import shutil
from typing import Annotated, Dict, Any, Optional

import torch
from google.cloud import aiplatform, storage as gcs_storage
from zenml import step, get_step_context
from omegaconf import DictConfig
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from ..serving.onnx_export import export_tft_to_onnx

logger = logging.getLogger(__name__)

# ---- Constants ----
SERVING_CONTAINER_URI = (
    "us-east1-docker.pkg.dev/realtime-headway-prediction/"
    "mlops-images/headway-serving:latest"
)
MODEL_DISPLAY_NAME = "headway-tft"


def _save_dataset_params(training_dataset: TimeSeriesDataSet, output_dir: str) -> str:
    """Persist the dataset parameters needed to reconstruct inputs at serving time.

    This includes categorical encoders (label mappings), normalizer params
    (center/scale per group), and the column ordering the model expects.
    """
    params = training_dataset.get_parameters()

    # Serialize categorical encoder mappings
    encoder_mappings = {}
    cat_encoders = params.get("categorical_encoders", {})
    for col_name, encoder in cat_encoders.items():
        if hasattr(encoder, "classes_"):
            # NaNLabelEncoder stores classes_ as a dict {value: int}
            encoder_mappings[col_name] = {
                str(k): int(v) for k, v in encoder.classes_.items()
            }
        elif hasattr(encoder, "mapping"):
            encoder_mappings[col_name] = {
                str(k): int(v) for k, v in encoder.mapping.items()
            }

    # Serialize target normalizer params (GroupNormalizer)
    #
    # GroupNormalizer stores fitted values in `norm_`, a DataFrame with
    # columns ["center", "scale"] indexed by the *encoded* group_id (int).
    # We map those back to original group names using the categorical encoder.
    normalizer_params = {}
    target_norm = training_dataset.target_normalizer
    if target_norm is not None and hasattr(target_norm, "norm_"):
        norm_df = target_norm.norm_

        # Build inverse mapping: encoded_int -> original_name
        gid_encoder = cat_encoders.get("group_id", cat_encoders.get("__group_id__group_id"))
        inv_map = {}
        if gid_encoder is not None and hasattr(gid_encoder, "classes_"):
            inv_map = {v: k for k, v in gid_encoder.classes_.items()}

        center_map = {}
        scale_map = {}
        for encoded_id, row in norm_df.iterrows():
            orig_name = str(inv_map.get(encoded_id, encoded_id))
            center_map[orig_name] = float(row["center"])
            scale_map[orig_name] = float(row["scale"])

        normalizer_params["center"] = center_map
        normalizer_params["scale"] = scale_map
        logger.info(
            "Serialized normalizer params: %d groups — %s",
            len(center_map),
            list(center_map.keys()),
        )
    elif target_norm is not None:
        logger.warning(
            "Target normalizer %s has no norm_ attribute — "
            "normalizer_params will be empty!",
            type(target_norm).__name__,
        )

    if not normalizer_params:
        logger.warning("normalizer_params is EMPTY — predictions will not be denormalized!")

    # Serialize per-feature scaler params (EncoderNormalizer with method="standard")
    scaler_params = {}
    dataset_scalers = getattr(training_dataset, "scalers", {}) or {}
    for col_name, scaler in dataset_scalers.items():
        if hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
            scaler_params[col_name] = {
                "center": float(scaler.center_),
                "scale": float(scaler.scale_),
            }
        elif hasattr(scaler, "parameters"):
            # Fallback: some EncoderNormalizer versions store a parameters dict
            sp = scaler.parameters or {}
            scaler_params[col_name] = {
                "center": float(sp.get("center", 0.0)),
                "scale": float(sp.get("scale", 1.0)),
            }
    if scaler_params:
        logger.info(
            "Serialized scaler params: %d features — %s",
            len(scaler_params),
            list(scaler_params.keys()),
        )
    else:
        logger.info("No per-feature scalers to serialize (raw inputs).")

    dataset_params = {
        "categorical_encoders": encoder_mappings,
        "normalizer_params": normalizer_params,
        "scaler_params": scaler_params,
    }

    path = os.path.join(output_dir, "dataset_params.json")
    with open(path, "w") as f:
        json.dump(dataset_params, f, indent=2)
    logger.info("Saved dataset parameters to %s", path)
    return path


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


@step(enable_cache=False)
def register_model(
    model: TemporalFusionTransformer,
    training_dataset: TimeSeriesDataSet,
    config: DictConfig,
    test_mae: float,
    test_smape: float,
) -> Annotated[str, "model_resource_name"]:
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

    Returns
    -------
    str
        The Vertex AI Model Registry resource name.
    """
    project = config.infra.project_id
    location = config.infra.location
    artifact_bucket = config.infra.artifact_bucket

    # Determine run ID for artifact versioning
    try:
        context = get_step_context()
        run_id = context.pipeline_run.name
    except Exception:
        import uuid
        run_id = f"local-{uuid.uuid4().hex[:8]}"

    gcs_model_uri = f"{artifact_bucket}/models/{run_id}"
    logger.info("Deploying model for run: %s", run_id)

    # ---- 1. Export model artifacts to temp dir ----
    local_dir = tempfile.mkdtemp(prefix="deploy_model_")
    try:
        # ONNX export
        logger.info("Exporting model to ONNX...")
        export_tft_to_onnx(
            tft_model=model,
            output_path=local_dir,
            encoder_length=config.processing.max_encoder_length,
            prediction_length=config.processing.max_prediction_length,
        )

        # Dataset parameters (encoders + normalizers)
        _save_dataset_params(training_dataset, local_dir)

        # Also save a PyTorch state_dict as backup
        torch.save(model.state_dict(), os.path.join(local_dir, "model_state_dict.pt"))
        logger.info("Saved PyTorch state_dict as backup")

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
            "format": "onnx",
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
