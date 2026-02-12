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

from mlops_pipeline.src.serving.onnx_export import export_tft_to_onnx

logger = logging.getLogger(__name__)

# ---- Constants ----
SERVING_CONTAINER_URI = (
    "us-east1-docker.pkg.dev/realtime-headway-prediction/"
    "mlops-images/headway-serving:latest"
)
ENDPOINT_DISPLAY_NAME = "headway-prediction-endpoint"
MODEL_DISPLAY_NAME = "headway-tft"
SERVING_MACHINE_TYPE = "n1-standard-4"
MIN_REPLICAS = 1
MAX_REPLICAS = 3


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
    normalizer_params = {}
    target_norm = params.get("target_normalizer", None)
    if target_norm is not None and hasattr(target_norm, "get_parameters"):
        try:
            norm_p = target_norm.get_parameters()
            # Convert numpy arrays / tensors to lists for JSON
            normalizer_params = {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in norm_p.items()
            }
        except Exception:
            pass
    elif target_norm is not None:
        # Fallback: store center_ and scale_ if available
        if hasattr(target_norm, "center_"):
            normalizer_params["center"] = {
                str(k): float(v) for k, v in target_norm.center_.items()
            }
        if hasattr(target_norm, "scale_"):
            normalizer_params["scale"] = {
                str(k): float(v) for k, v in target_norm.scale_.items()
            }

    dataset_params = {
        "categorical_encoders": encoder_mappings,
        "normalizer_params": normalizer_params,
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


def _get_or_create_endpoint(
    project: str, location: str
) -> aiplatform.Endpoint:
    """Get the existing prediction endpoint or create a new one."""
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"',
        project=project,
        location=location,
    )
    if endpoints:
        endpoint = endpoints[0]
        logger.info("Using existing endpoint: %s", endpoint.resource_name)
        return endpoint

    endpoint = aiplatform.Endpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        project=project,
        location=location,
    )
    logger.info("Created new endpoint: %s", endpoint.resource_name)
    return endpoint


@step(enable_cache=False)
def deploy_model(
    model: TemporalFusionTransformer,
    training_dataset: TimeSeriesDataSet,
    config: DictConfig,
    test_mae: float,
    test_smape: float,
) -> Annotated[str, "endpoint_resource_name"]:
    """Export model to ONNX, register in Vertex AI, and deploy to endpoint.

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
        The Vertex AI endpoint resource name.
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

        # ---- 4. Deploy to endpoint ----
        endpoint = _get_or_create_endpoint(project, location)

        # Undeploy existing models to do a clean swap
        if endpoint.traffic_split:
            logger.info("Undeploying previous model versions...")
            endpoint.undeploy_all()

        logger.info("Deploying model to endpoint...")
        endpoint.deploy(
            model=vertex_model,
            deployed_model_display_name=f"{MODEL_DISPLAY_NAME}-{run_id[:16]}",
            machine_type=SERVING_MACHINE_TYPE,
            min_replica_count=MIN_REPLICAS,
            max_replica_count=MAX_REPLICAS,
            traffic_percentage=100,
        )
        logger.info("Model deployed to endpoint: %s", endpoint.resource_name)

        return endpoint.resource_name

    finally:
        shutil.rmtree(local_dir, ignore_errors=True)
