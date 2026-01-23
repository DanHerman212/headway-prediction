from kfp import dsl
from kfp import compiler
from google_cloud_pipeline_components.types import artifact_types
import os

# Define the image URI - This expects the user to build the image containing the ml_pipelines package
TENSORFLOW_IMAGE_URI = os.environ.get("TENSORFLOW_IMAGE_URI", "gcr.io/your-project/headway-training:latest")

@dsl.container_component
def extract_bq_data(
    project_id: str,
    output_csv: dsl.Output[dsl.Dataset],
    query: str = "",
):
    """
    Component to extract data from BigQuery using DataExtractor.
    """
    return dsl.ContainerSpec(
        image=TENSORFLOW_IMAGE_URI,
        command=["python", "-m", "ml_pipelines.data.data"],
        args=[
            "--project_id", project_id,
            "--output_csv", output_csv.path,
            "--query", query
        ]
    )

@dsl.container_component
def preprocess_data(
    input_csv: dsl.Input[dsl.Dataset],
    output_csv: dsl.Output[dsl.Dataset],
):
    """
    Component to preprocess data using DataPreprocessor.
    """
    return dsl.ContainerSpec(
        image=TENSORFLOW_IMAGE_URI,
        command=["python", "-m", "ml_pipelines.data.preprocessing"],
        args=[
            "--input_csv", input_csv.path,
            "--output_csv", output_csv.path
        ]
    )


@dsl.container_component
def train_model(
    input_csv: dsl.Input[dsl.Dataset],
    model_dir: dsl.Output[artifact_types.UnmanagedContainerModel],
    test_dataset: dsl.Output[dsl.Dataset],
    epochs: int = 100,
):
    """
    Component to train the model using Trainer.
    """
    return dsl.ContainerSpec(
        image=TENSORFLOW_IMAGE_URI,
        command=["python", "-m", "ml_pipelines.training.train"],
        args=[
            "--input_csv", input_csv.path,
            "--model_dir", model_dir.path,
            "--test_dataset_path", test_dataset.path,
            "--epochs", str(epochs)
        ]
    )

@dsl.container_component
def evaluate_model(
    model_dir: dsl.Input[artifact_types.UnmanagedContainerModel],
    test_dataset: dsl.Input[dsl.Dataset],
    metrics: dsl.Output[dsl.Metrics],
    plots_dir: dsl.Output[dsl.Artifact],
):
    """
    Component to evaluate model using ModelEvaluator.
    """
    return dsl.ContainerSpec(
        image=TENSORFLOW_IMAGE_URI,
        command=["python", "-m", "ml_pipelines.evaluation.evaluate_model"],
        args=[
            "--model", model_dir.path,
            "--data", test_dataset.path,
            "--pre_split",
            "--output", plots_dir.path,
            "--metrics_output", metrics.path
        ]
    )


@dsl.pipeline(
    name="Headway Prediction Training Pipeline",
    description="End-to-end training pipeline for subway headway prediction"
)
def training_pipeline(
    project_id: str,
    epochs: int = 50,
):
    """
    Defines the training pipeline.
    
    Args:
        project_id: GCP Project ID for BigQuery access
        epochs: Number of training epochs
    """
    # 1. Extract
    extract_op = extract_bq_data(
        project_id=project_id,
    )
    
    # 3. Train
    train_op = train_model(
        input_csv=preprocess_op.outputs["output_csv"],
        epochs=epochs
    )
    
    # 4. Evaluate
    evaluate_op = evaluate_model(
        model_dir=train_op.outputs["model_dir"],
        test_dataset=train_op.outputs["test_dataset"]
    )
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="headway_pipeline.json"
    )
    print("Pipeline compiled to headway_pipeline.json")
