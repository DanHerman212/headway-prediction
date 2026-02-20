"""Retrieve eval artifacts from the latest Vertex AI Pipeline run.

Downloads the evaluation HTML plots and metrics from the most recent
successful headway-training-pipeline run stored in GCS.
"""

import argparse
import json
import os

from google.cloud import aiplatform, storage

PROJECT_ID = "realtime-headway-prediction"
LOCATION = "us-east1"
PIPELINE_ROOT = "gs://mlops-artifacts-realtime-headway-prediction/pipeline_runs"


def main():
    parser = argparse.ArgumentParser(description="Retrieve eval artifacts from latest pipeline run")
    parser.add_argument("--run-name", type=str, default=None, help="Specific pipeline run name (defaults to latest)")
    args = parser.parse_args()

    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Find the latest successful pipeline job
    jobs = aiplatform.PipelineJob.list(
        filter='display_name:"headway-training" AND state="PIPELINE_STATE_SUCCEEDED"',
        order_by="create_time desc",
    )
    if not jobs:
        print("No successful training pipeline runs found.")
        return

    job = jobs[0]
    print(f"Latest run: {job.display_name}")
    print(f"  Created: {job.create_time}")

    # Extract metrics and artifact URIs from the pipeline run
    task_details = job.task_details
    for task in task_details:
        if "evaluate" in task.task_name.lower():
            outputs = task.outputs or {}
            for output_name, output_detail in outputs.items():
                artifacts = output_detail.artifacts if hasattr(output_detail, "artifacts") else []
                for artifact in artifacts:
                    uri = artifact.uri
                    print(f"\n{output_name}:")
                    print(f"  URI: {uri}")

                    if uri.startswith("gs://") and uri.endswith(".html"):
                        local_path = f"/tmp/{output_name}.html"
                        _download_gcs(uri, local_path)
                        print(f"  Saved to: {local_path}")

            # Check for metrics
            if hasattr(task, "execution") and task.execution:
                metadata = task.execution.metadata or {}
                if "test_mae" in metadata:
                    print(f"\ntest_mae: {metadata['test_mae']}")
                if "test_smape" in metadata:
                    print(f"test_smape: {metadata['test_smape']}")


def _download_gcs(gcs_uri: str, local_path: str) -> None:
    """Download a GCS object to a local file."""
    bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
    blob_name = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(local_path)


if __name__ == "__main__":
    main()
