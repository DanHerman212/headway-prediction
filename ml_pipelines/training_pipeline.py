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
            # Ensure RUN_NAME is set for consistent directory structure
            if [ -z "$RUN_NAME" ]; then
                # Fallback if not passed by pipeline
                echo "Warning: RUN_NAME env var not set. Generating timestamp-based name."
                RUN_NAME="headway-model-$(date +%Y%m%d-%H%M%S)"
            else
                # Append random/timestamp to user-provided name to ensure uniqueness if they re-use the same string
                # or just accept it if they want to append to same run.
                # Here we assume unique runs per pipeline job:
                # But actually, the .env RUN_NAME is usually static. To separate runs we need dynamic suffix.
                # However, User asked why .env RUN_NAME is ignored. 
                # Pipelines passes parameters as arguments, not env vars usually, unless explicitly mapped.
                # But here we are inside the script args.
                echo "Using provided Run Name: $RUN_NAME"
                # If the user wants exact name, we keep it. If they want uniqueness, they should manage it.
                # But Vertex usually needs unique Folder per run for cleanest TB.
                # Let's append timestamp if the name looks static (e.g. doesn't end in numbers)
                if [[ ! "$RUN_NAME" =~ [0-9]{6} ]]; then
                     RUN_NAME="${RUN_NAME}-$(date +%Y%m%d-%H%M%S)"
                fi
            fi
            export RUN_NAME
            
            # 1. Train directly to GCS (Source of Truth)
            # We use the tensorboard_root (gs:// bucket) passed from pipeline
            # We append RUN_NAME to ensure we write to a unique subdirectory
            GCS_LOG_DIR="$TB_ROOT/$RUN_NAME"
            
            echo "Training with Run Name: $RUN_NAME"
            echo "Logs will be written to: $GCS_LOG_DIR"
            
            # Define function to sync and upload logs (to be called on exit)
            upload_logs() {
                echo "Initiating Post-Training Log Upload..."
                
                # 2. Sync Logs to Local Dir
                LOCAL_SYNC_DIR="/tmp/tb_sync"
                # Ensure we sync into a subdir matching run name so TB sees structure: root/run_name/events
                LOCAL_RUN_DIR="$LOCAL_SYNC_DIR"
                mkdir -p $LOCAL_RUN_DIR
                
                # Check if GCS_LOG_DIR is a gs:// path or a local/fuse path
                if [[ "$GCS_LOG_DIR" == gs://* ]]; then
                    echo "Downloading logs from GCS ($GCS_LOG_DIR) to local..."
                    
                    # Write Python script to file to avoid nesting quote issues
                    cat <<EOF_PY > /tmp/sync_logs.py
import os
import sys
from google.cloud import storage

def download_blob_folder(bucket_name, source_folder, destination_dir):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=source_folder))
        
        if not blobs:
            print(f'Warning: No files found in gs://{bucket_name}/{source_folder}')
            return

        print(f'Found {len(blobs)} files in {source_folder}')
        
        for blob in blobs:
            # Construct local path
            # blob.name includes full prefix (e.g. users/me/runs/run1/train/events...)
            # source_folder is (users/me/runs/run1)
            # relative is (train/events...)
            relative_path = os.path.relpath(blob.name, source_folder)
            
            # We want to preserve 'train/' or 'validation/' structure
            local_path = os.path.join(destination_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print(f'Downloaded: {local_path}')
            
    except Exception as e:
        print(f'Log Sync Error: {e}')
        # Log error but don't crash main process
        # sys.exit(0) 

# Get env vars
gcs_path = os.environ.get('GCS_LOG_DIR', '')
destination = os.environ.get('LOCAL_SYNC_DIR', '')

if gcs_path and destination:
    path_parts = gcs_path.replace('gs://', '').split('/')
    bucket_name = path_parts[0]
    prefix = '/'.join(path_parts[1:])
    download_blob_folder(bucket_name, prefix, destination)
else:
    print('Environment variables GCS_LOG_DIR or LOCAL_SYNC_DIR missing')
EOF_PY
                    # Run the sync script
                    export GCS_LOG_DIR
                    export LOCAL_SYNC_DIR="$LOCAL_RUN_DIR"
                    python /tmp/sync_logs.py
                    
                else:
                    # Assume Local or FUSE path
                    echo "Detailed check of log path: $GCS_LOG_DIR"
                    if [ -d "$GCS_LOG_DIR" ]; then
                        echo "Recursively copying logs from filesystem/FUSE..."
                        # Use cp -r
                        cp -r "$GCS_LOG_DIR"/* "$LOCAL_RUN_DIR"/ || echo "Copy warning: directory might be empty"
                    else
                        echo "Warning: Log directory $GCS_LOG_DIR does not exist on filesystem."
                    fi
                fi
                
                echo "Verifying Sync Directory Contents:"
                ls -R $LOCAL_SYNC_DIR
                
                # 3. Deterministic Upload
                # We upload the parent of the run dir to ensure the run name is key
                # BUT tb-gcp-uploader uploads the CONTENTS of --logdir.
                # So if logdir=/tmp/tb_sync, and it contains files, they are uploaded to root.
                # If we have subdirs /tmp/tb_sync/train, it works.
                
                echo "Starting Batch Upload to Vertex TensorBoard..."
                tb-gcp-uploader --tensorboard_resource_name $TB_RESOURCE \
                    --logdir $LOCAL_SYNC_DIR \
                    --experiment_name "headway-prediction-experiments" \
                    --one_shot=True
                
                echo "TensorBoard Upload Sequence Complete."
            }
            
            # TRAP exit to ensure upload happens even on failure
            trap upload_logs EXIT
            
            # Execute Training
            python -m ml_pipelines.training.train \
                --input_csv "$INPUT_CSV" \
                --model_dir "$MODEL_DIR" \
                --test_dataset_path "$TEST_DATASET_PATH" \
                --epochs "$EPOCHS" \
                --tensorboard_dir "$TB_ROOT" \
                --tensorboard_resource_name "$TB_RESOURCE"
            
            TRAIN_EXIT_CODE=$?
            
            echo "Training Script Exit Code: $TRAIN_EXIT_CODE"
            
            if [ $TRAIN_EXIT_CODE -ne 0 ]; then
                echo "Critical: Training failed. See logs above."
                exit $TRAIN_EXIT_CODE
            fi
            
            # Verify file creation (Debugging for GCS Fuse consistency)
            echo "Verifying test dataset creation..."
            if [ -d "$TEST_DATASET_PATH" ]; then
                echo "Listing $TEST_DATASET_PATH:"
                ls -la "$TEST_DATASET_PATH" || echo "Listing failed"
            else
                echo "Warning: $TEST_DATASET_PATH is not a directory or does not exist locally (via Fuse)."
            fi

            # Capture exit code (trap will fire after this)
            exit 0
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
    # train_op.set_accelerator_type("NVIDIA_TESLA_T4")
    train_op.set_accelerator_limit(1)
    # Ensure sufficient CPU/RAM for the A100 instance type (e.g., a2-highgpu-1g)
    train_op.set_cpu_limit("12")
    train_op.set_memory_limit("85G")
    
    # 4. Evaluate
    evaluate_op = evaluate_model(
        model_dir=train_op.outputs["model_dir"],
        test_dataset=train_op.outputs["test_dataset"]
    )
