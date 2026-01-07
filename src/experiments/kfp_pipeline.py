#!/usr/bin/env python3
"""
Kubeflow Pipeline for Vertex AI - Parallel Regularization Experiments

This is an alternative to vertex_pipeline.py that uses Kubeflow Pipelines (KFP)
for more sophisticated orchestration, including automatic result aggregation
and comparison.

Usage:
    python -m src.experiments.kfp_pipeline \
        --project your-gcp-project \
        --bucket your-gcs-bucket \
        --region us-central1
"""

import argparse
from datetime import datetime
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import component, Output, Artifact, Metrics
from google.cloud import aiplatform


# ============================================================================
# Pipeline Components
# ============================================================================

@component(
    base_image="python:3.10-slim",
    packages_to_install=["google-cloud-aiplatform>=1.38.0"],
)
def train_experiment(
    project: str,
    location: str,
    bucket: str,
    exp_id: int,
    exp_name: str,
    container_uri: str,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    tensorboard_id: str,
    service_account: str,
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [("job_name", str), ("output_dir", str)]):
    """
    Kubeflow component that submits and monitors a single training experiment.
    """
    from google.cloud import aiplatform
    from datetime import datetime
    import json
    
    # Initialize
    aiplatform.init(project=project, location=location)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"headway-exp{exp_id}-{exp_name}-{timestamp}"
    
    # GCS paths
    data_dir = f"gs://{bucket}/headway-prediction/data"
    output_dir = f"gs://{bucket}/headway-prediction/outputs/{timestamp}/exp_{exp_id:02d}_{exp_name}"
    
    # TensorBoard resource name
    tensorboard_resource = f"projects/{project}/locations/{location}/tensorboards/{tensorboard_id}"
    
    # Create and run job with TensorBoard integration
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=[
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
        ],
        staging_bucket=f"gs://{bucket}/staging",
    )
    
    # Run synchronously with TensorBoard streaming
    job.run(
        sync=True,
        tensorboard=tensorboard_resource,
        service_account=service_account,
    )
    
    # Log metrics
    metrics.log_metric("exp_id", exp_id)
    metrics.log_metric("job_state", str(job.state))
    
    # Return outputs for downstream components
    from collections import namedtuple
    outputs = namedtuple("Outputs", ["job_name", "output_dir"])
    return outputs(job_name=job_name, output_dir=output_dir)


@component(
    base_image="python:3.10-slim",
    packages_to_install=["google-cloud-storage>=2.0.0"],
)
def aggregate_results(
    bucket: str,
    exp1_output: str,
    exp2_output: str,
    exp3_output: str,
    exp4_output: str,
    metrics: Output[Metrics],
) -> str:
    """
    Aggregate results from all experiments and determine the best configuration.
    """
    from google.cloud import storage
    import json
    
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    
    results = []
    output_dirs = [exp1_output, exp2_output, exp3_output, exp4_output]
    
    for output_dir in output_dirs:
        # Parse GCS path
        path = output_dir.replace(f"gs://{bucket}/", "")
        results_path = f"{path}/results.json"
        
        try:
            blob = bucket_obj.blob(results_path)
            content = blob.download_as_text()
            result = json.loads(content)
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {results_path}: {e}")
            continue
    
    if not results:
        return "No results found"
    
    # Find best experiment by validation RMSE
    best = min(results, key=lambda x: x["results"]["best_val_rmse_seconds"])
    
    # Log comparison metrics
    for r in results:
        exp_id = r["exp_id"]
        metrics.log_metric(f"exp{exp_id}_val_rmse", r["results"]["best_val_rmse_seconds"])
        metrics.log_metric(f"exp{exp_id}_val_r2", r["results"]["best_val_r_squared"])
        metrics.log_metric(f"exp{exp_id}_best_epoch", r["results"]["best_epoch"])
    
    # Summary
    summary = {
        "best_experiment": best["exp_id"],
        "best_config": best["config"],
        "best_val_rmse": best["results"]["best_val_rmse_seconds"],
        "best_val_r2": best["results"]["best_val_r_squared"],
        "all_results": [
            {
                "exp_id": r["exp_id"],
                "exp_name": r["exp_name"],
                "val_rmse": r["results"]["best_val_rmse_seconds"],
                "val_r2": r["results"]["best_val_r_squared"],
            }
            for r in results
        ],
    }
    
    metrics.log_metric("best_experiment", best["exp_id"])
    metrics.log_metric("best_val_rmse", best["results"]["best_val_rmse_seconds"])
    
    return json.dumps(summary, indent=2)


# ============================================================================
# Pipeline Definition
# ============================================================================

@dsl.pipeline(
    name="headway-regularization-sweep",
    description="Parallel regularization experiments for ConvLSTM headway prediction",
)
def regularization_pipeline(
    project: str,
    location: str,
    bucket: str,
    tensorboard_id: str = "3732815588020453376",
    service_account: str = "156116751740-compute@developer.gserviceaccount.com",
    container_uri: str = "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest",
    machine_type: str = "a2-highgpu-1g",
    accelerator_type: str = "NVIDIA_TESLA_A100",
    accelerator_count: int = 1,
):
    """
    Pipeline that runs 4 regularization experiments in parallel and aggregates results.
    """
    
    # Experiment configurations
    experiments = [
        {"exp_id": 1, "exp_name": "baseline"},
        {"exp_id": 2, "exp_name": "dropout_only"},
        {"exp_id": 3, "exp_name": "dropout_l2"},
        {"exp_id": 4, "exp_name": "full_regularization"},
    ]
    
    # Submit all experiments in parallel
    exp_tasks = []
    for exp in experiments:
        task = train_experiment(
            project=project,
            location=location,
            bucket=bucket,
            exp_id=exp["exp_id"],
            exp_name=exp["exp_name"],
            container_uri=container_uri,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            tensorboard_id=tensorboard_id,
            service_account=service_account,
        )
        exp_tasks.append(task)
    
    # Aggregate results after all experiments complete
    aggregate_task = aggregate_results(
        bucket=bucket,
        exp1_output=exp_tasks[0].outputs["output_dir"],
        exp2_output=exp_tasks[1].outputs["output_dir"],
        exp3_output=exp_tasks[2].outputs["output_dir"],
        exp4_output=exp_tasks[3].outputs["output_dir"],
    )


# ============================================================================
# Pipeline Submission
# ============================================================================

def compile_and_run(
    project: str,
    location: str,
    bucket: str,
    container_uri: str,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    tensorboard_id: str = "3732815588020453376",
    service_account: str = "156116751740-compute@developer.gserviceaccount.com",
    pipeline_root: str = None,
):
    """Compile the pipeline and submit to Vertex AI Pipelines."""
    from kfp import compiler
    
    # Compile pipeline to JSON
    pipeline_file = "/tmp/regularization_pipeline.json"
    compiler.Compiler().compile(
        pipeline_func=regularization_pipeline,
        package_path=pipeline_file,
    )
    
    # Initialize Vertex AI
    aiplatform.init(project=project, location=location)
    
    # Set pipeline root for artifacts
    if pipeline_root is None:
        pipeline_root = f"gs://{bucket}/headway-prediction/pipeline_root"
    
    # Create and run pipeline job
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job = aiplatform.PipelineJob(
        display_name=f"headway-regularization-sweep-{timestamp}",
        template_path=pipeline_file,
        pipeline_root=pipeline_root,
        parameter_values={
            "project": project,
            "location": location,
            "bucket": bucket,
            "tensorboard_id": tensorboard_id,
            "service_account": service_account,
            "container_uri": container_uri,
            "machine_type": machine_type,
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
        },
    )
    
    print("=" * 60)
    print("Submitting Kubeflow Pipeline")
    print("=" * 60)
    print(f"Pipeline root: {pipeline_root}")
    print(f"TensorBoard: {tensorboard_id}")
    print("=" * 60)
    
    job.submit()
    
    print("\nPipeline submitted!")
    print(f"Job name: {job.display_name}")
    print(f"Resource: {job.resource_name}")
    print(f"\nView at: https://console.cloud.google.com/vertex-ai/pipelines/runs?project={project}")
    print(f"TensorBoard: https://console.cloud.google.com/vertex-ai/experiments/tensorboard-instances/regions/{location}/{tensorboard_id}?project={project}")
    
    return job


def main():
    parser = argparse.ArgumentParser(
        description="Run Kubeflow pipeline for regularization experiments"
    )
    parser.add_argument("--project", type=str, required=True, help="GCP project ID")
    parser.add_argument("--bucket", type=str, required=True, help="GCS bucket name")
    parser.add_argument("--region", type=str, default="us-central1", help="GCP region")
    parser.add_argument(
        "--container",
        type=str,
        default="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest",
        help="Container image URI",
    )
    parser.add_argument("--machine-type", type=str, default="a2-highgpu-1g")
    parser.add_argument("--accelerator-type", type=str, default="NVIDIA_TESLA_A100")
    parser.add_argument("--accelerator-count", type=int, default=1)
    
    args = parser.parse_args()
    
    compile_and_run(
        project=args.project,
        location=args.region,
        bucket=args.bucket,
        container_uri=args.container,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
    )


if __name__ == "__main__":
    main()
