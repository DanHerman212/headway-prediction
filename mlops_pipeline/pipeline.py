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

# Docker Settings
docker_settings = DockerSettings(
    requirements="mlops_pipeline/requirements.txt",
    replicate_local_python_environment=False
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
    hydra_overrides: Optional[List[str]] = None
):
    """
    End-to-end training pipeline for Headway Prediction.
    """
    # 1. Load Configuration (defaults from YAML, overrides from CLI)
    config = load_config_step(overrides=hydra_overrides)

    # 2. Ingest Data
    raw_df = ingest_data_step(file_path=data_path)

    # 3. Process Data (Returns training, val, test datasets)
    train_ds, val_ds, test_ds = process_data_step(
        raw_data=raw_df, 
        config=config
    )

    # 4. Train Model — GPU enabled via custom_job_parameters + ResourceSettings
    model = train_model_step.with_options(
        settings={
            "orchestrator.vertex": gpu_vertex_settings,
            "resources": ResourceSettings(gpu_count=1),
        },
    )(
        training_dataset=train_ds,
        validation_dataset=val_ds,
        config=config
    )

    # 5. Evaluate Model
    evaluate_model(
        model=model,
        test_dataset=test_ds,
        config=config
    )