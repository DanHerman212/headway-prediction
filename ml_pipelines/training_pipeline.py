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
    project_id: str,
    vertex_location: str,
    tensorboard_root: str,
    tensorboard_resource_name: str,
    run_name: str = "",
    epochs: int = 100,
):
    """
    Component to train the model using Trainer.
    """
    return dsl.ContainerSpec(
        image=TENSORFLOW_IMAGE_URI,
        command=["bash", "-c"],
        args=[
            '''
            export GCP_PROJECT_ID="$0"
            export VERTEX_LOCATION="$1"
            INPUT_CSV="$2"
            MODEL_DIR="$3"
            TEST_DATASET_PATH="$4"
            EPOCHS="$5"
            TB_ROOT="$6"
            TB_RESOURCE="$7"
            export RUN_NAME="$8"
            
            # Local log directory for TensorBoard uploader
            LOCAL_LOG_DIR="/tmp/tensorboard_logs"
            mkdir -p $LOCAL_LOG_DIR
            
            # Start TensorBoard Uploader in background
            # We explicitly tell it to watch the exact dir we write to
            echo "Starting TensorBoard Uploader..."
            tb-gcp-uploader --tensorboard_resource_name $TB_RESOURCE \
                --logdir $LOCAL_LOG_DIR \
                --experiment_name "headway-prediction-experiments" \
                --one_shot=False &
            
            UPLOADER_PID=$!
            
            # Run Training
            # We pass the LOCAL log dir to the training script so it writes events there
            python -m ml_pipelines.training.train \
                --input_csv "$INPUT_CSV" \
                --model_dir "$MODEL_DIR" \
                --test_dataset_path "$TEST_DATASET_PATH" \
                --epochs "$EPOCHS" \
                --tensorboard_dir "$LOCAL_LOG_DIR" \
                --tensorboard_resource_name "$TB_RESOURCE"
                
            TRAIN_EXIT_CODE=$?
            
            # Wait for final logs to upload (Histograms/Graphs can be large)
            echo "Training finished. Waiting 30s for TensorBoard logs to sync..."
            sleep 30
            
            # Cleanup
            kill $UPLOADER_PID || true
            
            # If we also want GCS backup of logs (optional, but good for persistence)
            # gsutil cp -r $LOCAL_LOG_DIR $TB_ROOT || true
            
            exit $TRAIN_EXIT_CODE
            ''',
            project_id,
            vertex_location,
            input_csv.path,
            model_dir.path,
            test_dataset.path,
            str(epochs),
            tensorboard_root,
            tensorboard_resource_name,
            run_name
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
    vertex_location: str,
    tensorboard_root: str,
    tensorboard_resource_name: str,
    run_name: str = "",
    epochs: int = 50,
):
    """
    Defines the training pipeline.
    
    Args:
        project_id: GCP Project ID for BigQuery access
        vertex_location: Vertex AI location (region)
        tensorboard_root: GCS path for TensorBoard logs
        tensorboard_resource_name: Resource Name of Managed TensorBoard
        run_name: Optional unique name for the experiment run
        epochs: Number of training epochs
    """
    # 1. Extract
    extract_op = extract_bq_data(
        project_id=project_id,
    )

    # 2. Preprocess
    preprocess_op = preprocess_data(
        input_csv=extract_op.outputs["output_csv"],
    )
    
    # 3. Train
    train_op = train_model(
        input_csv=preprocess_op.outputs["output_csv"],
        epochs=epochs,
        project_id=project_id,
        vertex_location=vertex_location,
        tensorboard_root=tensorboard_root,
        tensorboard_resource_name=tensorboard_resource_name,
        run_name=run_name
    )
    # Configure A100 GPU for training
    train_op.set_accelerator_type("NVIDIA_TESLA_A100")
    train_op.set_accelerator_type("NVIDIA_TESLA_A100")
    train_op.set_accelerator_limit(1)
    # Ensure sufficient CPU/RAM for the A100 instance type (e.g., a2-highgpu-1g)
    train_op.set_cpu_limit("12")
    train_op.set_memory_limit("85G")
    
    # 4. Evaluate
    evaluate_op = evaluate_model(
        model_dir=train_op.outputs["model_dir"],
        test_dataset=train_op.outputs["test_dataset"]
    )

