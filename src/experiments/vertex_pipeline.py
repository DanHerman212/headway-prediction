#!/usr/bin/env python3
"""
Vertex AI Pipeline for running parallel regularization experiments.

This pipeline submits 4 training jobs in parallel, each testing a different
regularization configuration. Results are saved to GCS and can be compared
in TensorBoard.

Usage:
    python -m src.experiments.vertex_pipeline \
        --project your-gcp-project \
        --bucket your-gcs-bucket \
        --region us-central1
"""

import argparse
import os
from datetime import datetime
from typing import List

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt


# Experiment configurations (matches experiment_config.py)
EXPERIMENTS = [
    {
        "exp_id": 1,
        "exp_name": "baseline",
        "spatial_dropout": 0.0,
        "weight_decay": 0.0,
        "learning_rate": 0.001,
    },
    {
        "exp_id": 2,
        "exp_name": "dropout_only",
        "spatial_dropout": 0.2,
        "weight_decay": 0.0,
        "learning_rate": 0.001,
    },
    {
        "exp_id": 3,
        "exp_name": "dropout_l2",
        "spatial_dropout": 0.2,
        "weight_decay": 0.0001,
        "learning_rate": 0.001,
    },
    {
        "exp_id": 4,
        "exp_name": "full_regularization",
        "spatial_dropout": 0.2,
        "weight_decay": 0.0001,
        "learning_rate": 0.0003,
    },
]


def create_training_job(
    project: str,
    location: str,
    bucket: str,
    exp_id: int,
    exp_name: str,
    container_uri: str,
    machine_type: str = "a2-highgpu-1g",
    accelerator_type: str = "NVIDIA_TESLA_A100",
    accelerator_count: int = 1,
) -> aiplatform.CustomJob:
    """
    Create a Vertex AI CustomJob for a single experiment.
    
    Args:
        project: GCP project ID
        location: GCP region (e.g., us-central1)
        bucket: GCS bucket name (without gs:// prefix)
        exp_id: Experiment ID (1-4)
        exp_name: Experiment name for display
        container_uri: Docker container URI with training code
        machine_type: Compute Engine machine type
        accelerator_type: GPU type
        accelerator_count: Number of GPUs
    
    Returns:
        Configured CustomJob (not yet submitted)
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"headway-exp{exp_id}-{exp_name}-{timestamp}"
    
    # GCS paths
    data_dir = f"gs://{bucket}/headway-prediction/data"
    output_dir = f"gs://{bucket}/headway-prediction/outputs/{timestamp}"
    
    # Worker pool spec
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_uri,
                "command": ["python", "-m", "src.experiments.run_experiment"],
                "args": [
                    f"--exp_id={exp_id}",
                    f"--data_dir={data_dir}",
                    f"--output_dir={output_dir}",
                ],
            },
        }
    ]
    
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        project=project,
        location=location,
        staging_bucket=f"gs://{bucket}/staging",
    )
    
    return job


def run_parallel_experiments(
    project: str,
    location: str,
    bucket: str,
    container_uri: str,
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
    experiment_ids: List[int] = None,
    sync: bool = False,
) -> List[aiplatform.CustomJob]:
    """
    Submit multiple experiments to run in parallel on Vertex AI.
    
    Args:
        project: GCP project ID
        location: GCP region
        bucket: GCS bucket name
        container_uri: Docker container with training code
        machine_type: VM type
        accelerator_type: GPU type
        accelerator_count: GPUs per job
        experiment_ids: Which experiments to run (default: all 4)
        sync: If True, wait for all jobs to complete
    
    Returns:
        List of submitted CustomJob objects
    """
    
    # Initialize Vertex AI
    aiplatform.init(project=project, location=location)
    
    # Filter experiments if specific IDs provided
    if experiment_ids:
        experiments_to_run = [e for e in EXPERIMENTS if e["exp_id"] in experiment_ids]
    else:
        experiments_to_run = EXPERIMENTS
    
    print("=" * 60)
    print("Submitting Parallel Training Experiments")
    print("=" * 60)
    print(f"Project: {project}")
    print(f"Location: {location}")
    print(f"Bucket: {bucket}")
    print(f"Container: {container_uri}")
    print(f"Machine: {machine_type} + {accelerator_count}x {accelerator_type}")
    print(f"Experiments: {[e['exp_id'] for e in experiments_to_run]}")
    print("=" * 60)
    
    jobs = []
    
    for exp in experiments_to_run:
        print(f"\nSubmitting Experiment {exp['exp_id']}: {exp['exp_name']}")
        print(f"  - spatial_dropout: {exp['spatial_dropout']}")
        print(f"  - weight_decay: {exp['weight_decay']}")
        print(f"  - learning_rate: {exp['learning_rate']}")
        
        job = create_training_job(
            project=project,
            location=location,
            bucket=bucket,
            exp_id=exp["exp_id"],
            exp_name=exp["exp_name"],
            container_uri=container_uri,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
        )
        
        # Submit job (non-blocking)
        # Note: run() submits the job and returns immediately when sync=False
        job.submit()
        jobs.append(job)
        
        print(f"  ✓ Job submitted: {job.display_name}")
        print(f"    Resource name: {job.resource_name}")
    
    print("\n" + "=" * 60)
    print(f"All {len(jobs)} experiments submitted!")
    print("=" * 60)
    print("\nMonitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
    print(f"\nTensorBoard logs: gs://{bucket}/headway-prediction/outputs/")
    
    if sync:
        print("\nWaiting for all jobs to complete...")
        for job in jobs:
            job.wait()
        print("All jobs completed!")
    
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel regularization experiments on Vertex AI"
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="GCP project ID"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="GCS bucket name (without gs:// prefix)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-central1",
        help="GCP region (default: us-central1)"
    )
    parser.add_argument(
        "--container",
        type=str,
        default=None,
        help="Container image URI. If not provided, uses default TF GPU image."
    )
    parser.add_argument(
        "--machine-type",
        type=str,
        default="a2-highgpu-1g",
        help="Compute Engine machine type (default: a2-highgpu-1g)"
    )
    parser.add_argument(
        "--accelerator-type",
        type=str,
        default="NVIDIA_TESLA_A100",
        help="GPU type (default: NVIDIA_TESLA_A100)"
    )
    parser.add_argument(
        "--accelerator-count",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)"
    )
    parser.add_argument(
        "--experiments",
        type=int,
        nargs="+",
        default=None,
        help="Experiment IDs to run (default: all 4). Example: --experiments 1 2"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Wait for all jobs to complete before exiting"
    )
    
    args = parser.parse_args()
    
    # Default container: TensorFlow GPU from Google's deep learning containers
    container_uri = args.container or "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest"
    
    run_parallel_experiments(
        project=args.project,
        location=args.region,
        bucket=args.bucket,
        container_uri=container_uri,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        experiment_ids=args.experiments,
        sync=args.sync,
    )


if __name__ == "__main__":
    main()
