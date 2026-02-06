from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.gcp.flavors.vertex_orchestrator_flavor import VertexOrchestratorSettings

from mlops_pipeline.src.steps.config_loader import load_config_step
from mlops_pipeline.src.steps.ingest_data import ingest_data_step
from mlops_pipeline.src.steps.process_data import process_data_step
from mlops_pipeline.src.steps.train_model import train_model_step
from mlops_pipeline.src.steps.evaluate_model import evaluate_model

# Define Vertex AI Settings (A100 GPU)
vertex_settings = VertexOrchestratorSettings(
    machine_type="a2-highgpu-1g",  # Contains 1x A100 GPU
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
)

# Define Docker Settings (Ensure dependencies are installed)
docker_settings = DockerSettings(
    requirements="mlops_pipeline/requirements.txt",
    replicate_local_python_environment=False
)

@pipeline(
    settings={
        "docker": docker_settings
    }
)
def headway_training_pipeline(
    data_path: str
):
    """
    End-to-end training pipeline for Headway Prediction.
    """
    # 1. Load Configuration
    config = load_config_step()

    # 2. Ingest Data
    raw_df = ingest_data_step(file_path=data_path)

    # 3. Process Data (Returns training, val, test datasets)
    train_ds, val_ds, test_ds = process_data_step(
        raw_data=raw_df, 
        config=config
    )

    # 4. Train Model (Apply GPU Settings Here)
    model = train_model_step.with_options(
        settings={"orchestrator.vertex": vertex_settings}
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