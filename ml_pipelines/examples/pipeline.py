from kfp import dsl
from kfp import compiler
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Artifact,
)
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
import os

# Get image URI from environment variable (injected by deploy script)
TENSORFLOW_IMAGE_URI = os.environ.get("TENSORFLOW_IMAGE_URI")
SERVING_IMAGE_URI = os.environ.get("SERVING_IMAGE_URI")

if not TENSORFLOW_IMAGE_URI or not SERVING_IMAGE_URI:
    raise ValueError("Image URIs must be set via environment variables (TENSORFLOW_IMAGE_URI, SERVING_IMAGE_URI)")

# 1. Component: Extract Data from BigQuery
@dsl.container_component
def extract_bq_data(
    project_id: str,
    query: str,
    output_dataset: dsl.Output[dsl.Dataset]
):
    return dsl.ContainerSpec(
        image=TENSORFLOW_IMAGE_URI,
        command=["python", "src/extract.py"],
        args=[
            "--project_id", project_id,
            "--query", query,
            "--output_csv", output_dataset.path
        ]
    )

# 2. Component Definition for Custom Scripts
# We define container components that use the custom image directly.

@dsl.container_component
def preprocess_component(
    input_csv: dsl.Input[dsl.Dataset],
    output_csv: dsl.Output[dsl.Dataset],
):
    return dsl.ContainerSpec(
        image=TENSORFLOW_IMAGE_URI,
        command=["python", "src/preprocess.py"],
        args=[
            "--input_csv", input_csv.path,
            "--output_csv", output_csv.path
        ]
    )

@dsl.container_component
def train_gru_component(
    input_csv: dsl.Input[dsl.Dataset],
    model_dir: dsl.Output[artifact_types.UnmanagedContainerModel],
    test_dataset: dsl.Output[dsl.Dataset],
):
    return dsl.ContainerSpec(
        image=TENSORFLOW_IMAGE_URI,
        command=["python", "src/train_gru.py"],
        args=[
            "--input_csv", input_csv.path,
            "--model_dir", model_dir.path,
            "--test_dataset_path", test_dataset.path
        ]
    )

@dsl.container_component
def evaluate_gru_component(
    test_dataset: dsl.Input[dsl.Dataset],
    model_dir: dsl.Input[artifact_types.UnmanagedContainerModel],
    input_csv: dsl.Input[dsl.Dataset],
    metrics: dsl.Output[dsl.Metrics],
    loss_plot: dsl.Output[dsl.HTML],
    prediction_plot: dsl.Output[dsl.HTML],
):
    return dsl.ContainerSpec(
        image=TENSORFLOW_IMAGE_URI,
        command=["python", "src/evaluate_gru.py"],
        args=[
            "--test_dataset_path", test_dataset.path,
            "--model_dir", model_dir.path,
            "--input_csv", input_csv.path,
            "--metrics_output_path", metrics.path,
            "--plot_output_path", loss_plot.path,
            "--prediction_plot_path", prediction_plot.path
        ]
    )

# Helper component to attach serving metadata
@dsl.component(base_image="python:3.10", packages_to_install=["google-cloud-pipeline-components"])
def attach_serving_spec(
    original_model: dsl.Input[artifact_types.UnmanagedContainerModel],
    model_with_spec: dsl.Output[artifact_types.UnmanagedContainerModel],
    serving_image_uri: str
):
    model_with_spec.uri = original_model.uri
    model_with_spec.metadata = {
        "containerSpec": {
            "imageUri": serving_image_uri,
            "healthRoute": "/health",
            "predictRoute": "/predict"
        }
    }

# 3. Pipeline Definition
@dsl.pipeline(
    name="forecasting-training-pipeline",
    description="Pipeline to extract data, preprocess, and train GRU model."
)
def forecasting_pipeline(
    project_id: str,
    bq_query: str,
    region: str = "us-east1",
    model_display_name: str = "gru-model-v1"
):
    # Step 1: Extract
    extract_task = extract_bq_data(
        project_id=project_id,
        query=bq_query
    )
    
    # Step 2: Preprocess
    preprocess_task = preprocess_component(
        input_csv=extract_task.outputs["output_dataset"]
    )
    # Disable caching to ensure we pick up the latest code changes in the image
    preprocess_task.set_caching_options(False)
    
    # Step 3: Train GRU
    train_gru_task = train_gru_component(
        input_csv=preprocess_task.outputs["output_csv"]
    )

    # Configure GPU resources
    train_gru_task.set_cpu_limit('4')
    train_gru_task.set_memory_limit('16G')
    train_gru_task.set_gpu_limit(1)
    train_gru_task.set_accelerator_type('NVIDIA_TESLA_T4')

    # Step 3.5: Attach Serving Spec (GRU)
    # We attach the serving container image URI to the model artifact metadata
    # so that Vertex AI knows which image to use for deployment.
    model_with_metadata_task = attach_serving_spec(
        original_model=train_gru_task.outputs["model_dir"],
        serving_image_uri=SERVING_IMAGE_URI
    )

    # Step 4: Upload to Model Registry (GRU)
    model_upload_task = ModelUploadOp(
        project=project_id,
        location=region,
        display_name=model_display_name,
        unmanaged_container_model=model_with_metadata_task.outputs["model_with_spec"],
    )

    # Step 5: Evaluate GRU
    evaluate_gru_task = evaluate_gru_component(
        test_dataset=train_gru_task.outputs["test_dataset"],
        model_dir=train_gru_task.outputs["model_dir"],
        input_csv=preprocess_task.outputs["output_csv"]
    )
    # Assign GPU to evaluation task to support CudnnRNNV3 ops
    evaluate_gru_task.set_cpu_limit('4')
    evaluate_gru_task.set_memory_limit('16G')
    evaluate_gru_task.set_gpu_limit(1)
    evaluate_gru_task.set_accelerator_type('NVIDIA_TESLA_T4')
    
    # Step 7: Create Endpoint
    endpoint_create_task = EndpointCreateOp(
        project=project_id,
        location=region,
        display_name="gru-endpoint",
    )

    # Step 8: Deploy GRU Model to Endpoint
    model_deploy_task = ModelDeployOp(
        endpoint=endpoint_create_task.outputs["endpoint"],
        model=model_upload_task.outputs["model"],
        deployed_model_display_name="gru-deployed",
        dedicated_resources_machine_type="n1-standard-2",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=forecasting_pipeline,
        package_path="forecasting_pipeline.json"
    )
