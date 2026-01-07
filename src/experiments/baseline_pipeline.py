#!/usr/bin/env python3
"""
Simplified Kubeflow Pipeline for Baseline Training

Single-job pipeline for the paper-faithful baseline ConvLSTM.
Designed for rapid iteration with full TensorBoard integration.

Usage:
    # Submit pipeline
    python -m src.experiments.baseline_pipeline

    # With custom TensorBoard
    python -m src.experiments.baseline_pipeline --tensorboard_id YOUR_ID
"""

import argparse
from datetime import datetime

from kfp import dsl
from kfp import compiler
from google.cloud import aiplatform


# ============================================================================
# Configuration
# ============================================================================

PROJECT = "time-series-478616"
REGION = "us-east1"
BUCKET = "st-convnet-training-configuration"
PIPELINE_ROOT = f"gs://{BUCKET}/pipeline-runs"

# Container image
TRAINING_IMAGE = f"gcr.io/{PROJECT}/headway-trainer:latest"

# Service account for TensorBoard access
SERVICE_ACCOUNT = f"vertex-ai-sa@{PROJECT}.iam.gserviceaccount.com"


# ============================================================================
# Training Component
# ============================================================================

@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=[]  # All deps in container
)
def train_baseline(
    epochs: int,
    batch_size: int,
    data_gcs_path: str,
    output_gcs_path: str,
) -> str:
    """
    Train paper-faithful baseline ConvLSTM.
    
    Returns:
        Path to results JSON on GCS
    """
    import subprocess
    import sys
    
    # Run training script
    cmd = [
        sys.executable, "-m", "src.experiments.run_baseline",
        "--data_dir", data_gcs_path,
        "--output_dir", output_gcs_path,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    return f"{output_gcs_path}/results.json"


# ============================================================================
# Pipeline Definition
# ============================================================================

@dsl.pipeline(
    name="baseline-convlstm-training",
    description="Train paper-faithful ConvLSTM baseline with full TensorBoard tracking"
)
def baseline_pipeline(
    epochs: int = 100,
    batch_size: int = 32,
    data_gcs_path: str = f"gs://{BUCKET}/data",
):
    """
    Single baseline training run.
    
    Args:
        epochs: Training epochs (paper default: 100)
        batch_size: Batch size (paper default: 32)
        data_gcs_path: GCS path to training data
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = f"gs://{BUCKET}/runs/baseline-{timestamp}"
    
    train_task = train_baseline(
        epochs=epochs,
        batch_size=batch_size,
        data_gcs_path=data_gcs_path,
        output_gcs_path=output_path,
    )
    
    # GPU configuration - A100 for fast training
    train_task.set_accelerator_type("NVIDIA_TESLA_A100")
    train_task.set_accelerator_count(1)
    train_task.set_cpu_limit("8")
    train_task.set_memory_limit("32G")


# ============================================================================
# Pipeline Submission
# ============================================================================

def compile_pipeline(output_path: str = "baseline_pipeline.json"):
    """Compile the pipeline to JSON."""
    compiler.Compiler().compile(
        pipeline_func=baseline_pipeline,
        package_path=output_path
    )
    print(f"Pipeline compiled to: {output_path}")
    return output_path


def submit_pipeline(
    epochs: int = 100,
    batch_size: int = 32,
    tensorboard_id: str = None,
    enable_caching: bool = False,
):
    """
    Submit pipeline to Vertex AI.
    
    Args:
        epochs: Training epochs
        batch_size: Batch size
        tensorboard_id: TensorBoard instance ID (optional)
        enable_caching: Enable KFP caching
    """
    # Initialize Vertex AI
    aiplatform.init(project=PROJECT, location=REGION)
    
    # Compile pipeline
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pipeline_file = f"/tmp/baseline_pipeline_{timestamp}.json"
    compile_pipeline(pipeline_file)
    
    # Pipeline parameters
    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "data_gcs_path": f"gs://{BUCKET}/data",
    }
    
    # Create job
    display_name = f"baseline-convlstm-{timestamp}"
    
    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=pipeline_file,
        pipeline_root=PIPELINE_ROOT,
        parameter_values=params,
        enable_caching=enable_caching,
    )
    
    print(f"\nSubmitting pipeline: {display_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    # Submit with optional TensorBoard
    submit_kwargs = {
        "service_account": SERVICE_ACCOUNT,
    }
    
    if tensorboard_id:
        tensorboard_resource = f"projects/{PROJECT}/locations/{REGION}/tensorboards/{tensorboard_id}"
        submit_kwargs["create_request_timeout"] = 600
        # Note: TensorBoard integration requires the training code to write to the correct log dir
        print(f"  TensorBoard: {tensorboard_id}")
    
    job.submit(**submit_kwargs)
    
    print(f"\nPipeline submitted!")
    print(f"Job: {job.resource_name}")
    print(f"\nMonitor at:")
    print(f"  https://console.cloud.google.com/vertex-ai/pipelines/runs?project={PROJECT}")
    
    return job


def submit_custom_job(
    epochs: int = 100,
    batch_size: int = 32,
    tensorboard_id: str = None,
):
    """
    Submit as Custom Training Job (simpler than KFP for single job).
    
    This is faster to start and easier to debug than full KFP.
    """
    aiplatform.init(project=PROJECT, location=REGION)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    display_name = f"baseline-convlstm-{timestamp}"
    output_path = f"gs://{BUCKET}/runs/baseline-{timestamp}"
    
    # Training command
    args = [
        "--data_dir", f"gs://{BUCKET}/data",
        "--output_dir", output_path,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]
    
    # Create custom job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=TRAINING_IMAGE,
        command=["python", "-m", "src.experiments.run_baseline"],
    )
    
    # TensorBoard integration
    tensorboard = None
    if tensorboard_id:
        tensorboard = f"projects/{PROJECT}/locations/{REGION}/tensorboards/{tensorboard_id}"
    
    print(f"\nSubmitting Custom Training Job: {display_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output: {output_path}")
    if tensorboard:
        print(f"  TensorBoard: {tensorboard_id}")
    
    # Run
    job.run(
        args=args,
        replica_count=1,
        machine_type="a2-highgpu-1g",  # A100 GPU
        accelerator_type="NVIDIA_TESLA_A100",
        accelerator_count=1,
        service_account=SERVICE_ACCOUNT,
        tensorboard=tensorboard,
        sync=False,  # Don't wait for completion
    )
    
    print(f"\nJob submitted!")
    print(f"\nMonitor at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT}")
    
    return job


def main():
    parser = argparse.ArgumentParser(
        description="Submit baseline training to Vertex AI"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100 per paper)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32 per paper)"
    )
    parser.add_argument(
        "--tensorboard_id",
        type=str,
        default=None,
        help="TensorBoard instance ID"
    )
    parser.add_argument(
        "--custom_job",
        action="store_true",
        help="Use CustomTrainingJob instead of KFP (simpler)"
    )
    parser.add_argument(
        "--compile_only",
        action="store_true",
        help="Only compile pipeline, don't submit"
    )
    
    args = parser.parse_args()
    
    if args.compile_only:
        compile_pipeline("baseline_pipeline.json")
        return
    
    if args.custom_job:
        submit_custom_job(
            epochs=args.epochs,
            batch_size=args.batch_size,
            tensorboard_id=args.tensorboard_id,
        )
    else:
        submit_pipeline(
            epochs=args.epochs,
            batch_size=args.batch_size,
            tensorboard_id=args.tensorboard_id,
        )


if __name__ == "__main__":
    main()
