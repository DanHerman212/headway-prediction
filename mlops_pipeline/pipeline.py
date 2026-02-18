from typing import List, Optional

from zenml import pipeline
from zenml.config import DockerSettings, ResourceSettings
from zenml.integrations.gcp.flavors.vertex_orchestrator_flavor import VertexOrchestratorSettings
from zenml.integrations.gcp.vertex_custom_job_parameters import VertexCustomJobParameters

from .src.steps.config_loader import load_config_step
from .src.steps.ingest_data import ingest_data_step
from .src.steps.process_data import process_data_step
from .src.steps.train_model import train_model_step
from .src.steps.evaluate_model import evaluate_model
from .src.steps.deploy_model import register_model
from .src.steps.fetch_best_vizier_params import fetch_best_vizier_params

# Docker Settings — use CUDA-enabled parent image for GPU support
docker_settings = DockerSettings(
    parent_image="pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime",
    requirements="mlops_pipeline/requirements.txt",
    replicate_local_python_environment=False,
    python_package_installer="pip",
)

# GPU settings for the training step
gpu_vertex_settings = VertexOrchestratorSettings(
    custom_job_parameters=VertexCustomJobParameters(
        machine_type="a2-highgpu-1g",
        accelerator_type="NVIDIA_TESLA_A100",
        accelerator_count=1,
    )
)

@pipeline(
    enable_cache=False,
    settings={
        "docker": docker_settings
    }
)
def headway_training_pipeline(
    data_path: str,
    hydra_overrides: Optional[List[str]] = None,
    use_vizier_params: bool = False,
):
    """
    End-to-end training pipeline for Headway Prediction.

    When ``use_vizier_params=True``, the best hyperparameters from the
    latest Vizier study are fetched and applied to the config inside
    the training step via OmegaConf.update.  Hydra defaults load first,
    then Vizier params override the model/training keys.
    """
    # 1. Load Configuration (defaults from YAML, overrides from CLI)
    config = load_config_step(overrides=hydra_overrides)

    # 1b. Optionally fetch best params from Vizier
    vizier_params = None
    if use_vizier_params:
        vizier_params = fetch_best_vizier_params(
            config=config,
        )

    # 2. Ingest Data
    raw_df = ingest_data_step(file_path=data_path)

    # 3. Process Data (Returns training, val, test datasets + time anchor)
    train_ds, val_ds, test_ds, time_anchor_iso = process_data_step(
        raw_data=raw_df, 
        config=config
    )

    # 4. Train Model — GPU enabled via custom_job_parameters + ResourceSettings
    #    Vizier params (if any) get applied to config inside the step
    model = train_model_step.with_options(
        settings={
            "orchestrator.vertex": gpu_vertex_settings,
            "resources": ResourceSettings(gpu_count=1),
        },
    )(
        training_dataset=train_ds,
        validation_dataset=val_ds,
        config=config,
        vizier_params=vizier_params,
    )

    # 5. Evaluate Model
    test_mae, test_smape, rush_hour_html, interpretation_html = evaluate_model(
        model=model,
        test_dataset=test_ds,
        config=config,
        time_anchor_iso=time_anchor_iso,
    )

    # 6. Register Model in Vertex AI Model Registry
    register_model(
        model=model,
        training_dataset=train_ds,
        config=config,
        test_mae=test_mae,
        test_smape=test_smape,
    )