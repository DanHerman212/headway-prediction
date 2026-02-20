"""
run.py — CLI entrypoint for Headway Prediction Pipelines on Vertex AI.

Compiles the KFP v2 pipeline to JSON and submits it as a Vertex AI PipelineJob.
Optionally triggers a Cloud Build to rebuild the training container image
with Docker layer caching (--build flag).
"""

import json
import os
import re
import subprocess
import sys
import argparse
import tempfile
import uuid
from datetime import datetime

# Ensure the project root is always on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import aiplatform
from kfp import compiler

from mlops_pipeline.pipeline import headway_training_pipeline, TRAINING_IMAGE
from mlops_pipeline.hpo_pipeline import headway_hpo_pipeline

# --------------- Constants ---------------------------------------------------
PROJECT_ID = "realtime-headway-prediction"
LOCATION = "us-east1"
PIPELINE_ROOT = "gs://mlops-artifacts-realtime-headway-prediction/pipeline_runs"
SERVICE_ACCOUNT = None  # Uses default compute SA; set explicitly if needed
CLOUDBUILD_CONFIG = "infra/cloudbuild_training.yaml"


def _sanitize_name(name: str) -> str:
    """Vertex AI experiment/run names: lowercase, [a-z0-9-], 128 chars max."""
    return re.sub(r"[^a-z0-9-]", "-", name.strip().lower())[:128].rstrip("-")


def _build_training_image() -> None:
    """Trigger Cloud Build to rebuild the training image with layer caching.

    Docker layer caching means:
      - Source-only changes → only the final COPY layer rebuilds (~10-15s)
      - requirements.txt changes → pip install layer rebuilds (~3-5 min)
    """
    print(f"\n{'='*60}")
    print(f"Building training image: {TRAINING_IMAGE}")
    print(f"{'='*60}\n")

    cmd = [
        "gcloud", "builds", "submit",
        "--project", PROJECT_ID,
        "--region", LOCATION,
        "--config", CLOUDBUILD_CONFIG,
        "--substitutions", f"_IMAGE_URI={TRAINING_IMAGE}",
        ".",
    ]
    print(f"$ {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nCloud Build failed (exit code {result.returncode}).")
        sys.exit(result.returncode)

    print(f"\nImage built and pushed: {TRAINING_IMAGE}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Headway Prediction Pipelines on Vertex AI")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["training", "hpo"],
        default="training",
        help="Pipeline to run: 'training' (standard) or 'hpo' (hyperparameter optimization)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="gs://mlops-artifacts-realtime-headway-prediction/data/training_data.parquet",
        help="Path to the training data parquet file",
    )
    parser.add_argument(
        "--use-vizier-params",
        action="store_true",
        default=False,
        help="Fetch best hyperparameters from the latest Vizier study.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Compile pipeline JSON only (do not submit to Vertex AI).",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        default=False,
        help="Rebuild the training container image via Cloud Build before submitting. "
             "Uses Docker layer caching — source-only changes take ~10-15s.",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        default=False,
        help="Build the image (if --build) but do not submit the pipeline.",
    )

    args, hydra_overrides = parser.parse_known_args()

    # ---- Build image if requested -------------------------------------------
    if args.build:
        _build_training_image()
        if args.no_submit:
            print("--no-submit flag set — skipping pipeline submission.")
            return

    # ---- Derive experiment and run names ------------------------------------
    experiment_name = "headway-tft"
    for override in hydra_overrides:
        if override.startswith("experiment_name="):
            experiment_name = override.split("=")[1]
            break
    experiment_name = _sanitize_name(experiment_name)

    timestamp = datetime.now(tz=None).strftime("%Y%m%d-%H%M%S")
    run_name = _sanitize_name(f"{experiment_name}-{timestamp}-{uuid.uuid4().hex[:6]}")

    # Ensure absolute path for local data
    data_path = args.data_path
    if not data_path.startswith("gs://"):
        data_path = os.path.abspath(data_path)

    # JSON-encode Hydra overrides for pipeline parameter
    overrides_json = json.dumps(hydra_overrides) if hydra_overrides else "null"

    print(f"Mode:            {args.mode.upper()}")
    print(f"Data:            {data_path}")
    print(f"Experiment:      {experiment_name}")
    print(f"Run name:        {run_name}")
    print(f"Hydra overrides: {hydra_overrides or '(none)'}")
    if args.use_vizier_params:
        print("Vizier param injection: ENABLED")

    # ---- Compile pipeline ---------------------------------------------------
    if args.mode == "training":
        pipeline_func = headway_training_pipeline
        pipeline_params = {
            "data_path": data_path,
            "run_name": run_name,
            "hydra_overrides_json": overrides_json,
            "use_vizier_params": args.use_vizier_params,
        }
    else:
        pipeline_func = headway_hpo_pipeline
        pipeline_params = {
            "data_path": data_path,
            "hydra_overrides_json": overrides_json,
        }

    pipeline_json = os.path.join(
        tempfile.gettempdir(), f"headway_{args.mode}_{run_name}.json"
    )
    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=pipeline_json,
    )
    print(f"Compiled pipeline to: {pipeline_json}")

    if args.local:
        print("--local flag set — skipping Vertex AI submission.")
        return

    # ---- Submit to Vertex AI ------------------------------------------------
    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
    )

    job = aiplatform.PipelineJob(
        display_name=f"headway-{args.mode}-{run_name}",
        template_path=pipeline_json,
        pipeline_root=PIPELINE_ROOT,
        parameter_values=pipeline_params,
        enable_caching=False,
    )

    print("Submitting pipeline to Vertex AI…")
    job.submit(
        service_account=SERVICE_ACCOUNT,
        experiment=experiment_name,
    )
    print(f"Pipeline job submitted: {job.resource_name}")
    print(f"Console: https://console.cloud.google.com/vertex-ai/pipelines/runs/"
          f"{job.resource_name.split('/')[-1]}?project={PROJECT_ID}")


if __name__ == "__main__":
    main()