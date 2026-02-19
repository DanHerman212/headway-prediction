#!/usr/bin/env python3
"""
deploy_endpoint.py
------------------
Standalone script to deploy the latest registered headway-tft model
to a Vertex AI Prediction Endpoint with Model Monitoring.

Steps:
  1. Find the latest 'headway-tft' model in Vertex AI Model Registry
  2. Create or reuse the prediction endpoint
  3. Deploy the model to the endpoint
  4. Generate training data stats baseline for monitoring
  5. Create a Model Monitoring job (feature drift + skew detection)
  6. Smoke test with a synthetic prediction request

Usage:
  python scripts/deploy_endpoint.py
  python scripts/deploy_endpoint.py --skip-monitoring   # deploy only, no monitoring
  python scripts/deploy_endpoint.py --dry-run            # print what would happen
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
from google.cloud import aiplatform
from google.cloud import storage as gcs_storage
from google.protobuf import json_format

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
PROJECT_ID = "realtime-headway-prediction"
LOCATION = "us-east1"
MODEL_DISPLAY_NAME = "headway-tft"
ENDPOINT_DISPLAY_NAME = "headway-prediction-endpoint"
ARTIFACT_BUCKET = "gs://mlops-artifacts-realtime-headway-prediction"
TRAINING_DATA_URI = f"{ARTIFACT_BUCKET}/data/training_data.parquet"
MONITORING_DATASET_URI = f"{ARTIFACT_BUCKET}/monitoring/training_baseline.csv"
PREDICT_SCHEMA_URI = f"{ARTIFACT_BUCKET}/monitoring/predict_instance_schema.yaml"
ANALYSIS_SCHEMA_URI = f"{ARTIFACT_BUCKET}/monitoring/analysis_instance_schema.yaml"

# Local schema files (checked into repo under infra/schemas/)
_SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "..", "infra", "schemas")

# Machine for serving — ONNX on CPU is plenty fast for single-step predictions
MACHINE_TYPE = "n1-standard-4"

# Features the model sees at serving time (numeric only for drift monitoring)
# These must match the feature columns the predictor.py sends to the ONNX model.
MONITORED_FEATURES = [
    "service_headway",
    "preceding_train_gap",
    "upstream_headway_14th",
    "travel_time_14th",
    "travel_time_14th_deviation",
    "travel_time_23rd",
    "travel_time_23rd_deviation",
    "travel_time_34th",
    "travel_time_34th_deviation",
    "stops_at_23rd",
    "hour_sin",
    "hour_cos",
    "empirical_median",
    "day_of_week",
]

# Default drift threshold per feature (Jensen-Shannon divergence)
DEFAULT_DRIFT_THRESHOLD = 0.3

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _find_latest_model() -> aiplatform.Model:
    """Find the most recently created 'headway-tft' model in the registry."""
    models = aiplatform.Model.list(
        filter=f'display_name="{MODEL_DISPLAY_NAME}"',
        order_by="create_time desc",
    )
    if not models:
        logger.error(
            "No model found with display_name='%s'. "
            "Run the training pipeline first (python mlops_pipeline/run.py --mode training).",
            MODEL_DISPLAY_NAME,
        )
        sys.exit(1)

    latest = models[0]
    logger.info(
        "Found model: %s (created %s)",
        latest.resource_name,
        latest.create_time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    # Log labels (test metrics, run_id, etc.)
    if latest.labels:
        for k, v in latest.labels.items():
            logger.info("  label: %s = %s", k, v)
    return latest


def _get_or_create_endpoint() -> aiplatform.Endpoint:
    """Reuse existing endpoint or create a new one."""
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"',
    )
    if endpoints:
        endpoint = endpoints[0]
        logger.info("Reusing existing endpoint: %s", endpoint.resource_name)
        return endpoint

    logger.info("Creating new endpoint: %s", ENDPOINT_DISPLAY_NAME)
    endpoint = aiplatform.Endpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        project=PROJECT_ID,
        location=LOCATION,
    )
    logger.info("Created endpoint: %s", endpoint.resource_name)
    return endpoint


def _deploy_model(
    endpoint: aiplatform.Endpoint,
    model: aiplatform.Model,
    dry_run: bool = False,
) -> None:
    """Deploy the model to the endpoint, replacing any existing deployment."""
    # Check if model is already deployed
    deployed_models = endpoint.gca_resource.deployed_models
    if deployed_models:
        existing_ids = [dm.id for dm in deployed_models]
        logger.info(
            "Endpoint has %d existing deployment(s): %s",
            len(existing_ids),
            existing_ids,
        )
        if not dry_run:
            logger.info("Undeploying existing model(s) before deploying new version...")
            for dm in deployed_models:
                endpoint.undeploy(deployed_model_id=dm.id)
                logger.info("  Undeployed %s", dm.id)

    if dry_run:
        logger.info("[DRY RUN] Would deploy model %s to endpoint %s",
                     model.resource_name, endpoint.resource_name)
        return

    logger.info(
        "Deploying model to endpoint (machine_type=%s)...",
        MACHINE_TYPE,
    )
    endpoint.deploy(
        model=model,
        deployed_model_display_name=f"{MODEL_DISPLAY_NAME}-serving",
        machine_type=MACHINE_TYPE,
        min_replica_count=1,
        max_replica_count=1,  # scale to 1 for now; increase when traffic warrants
        traffic_percentage=100,
        enable_access_logging=True,
    )
    logger.info("Model deployed successfully.")


def _generate_monitoring_baseline(dry_run: bool = False) -> str:
    """Generate a training feature baseline CSV for Model Monitoring.

    Downloads the training parquet, extracts the monitored feature columns,
    and uploads a CSV to GCS. Returns the GCS URI of the baseline CSV.
    """
    logger.info("Generating monitoring baseline from %s", TRAINING_DATA_URI)

    if dry_run:
        logger.info("[DRY RUN] Would generate baseline CSV at %s", MONITORING_DATASET_URI)
        return MONITORING_DATASET_URI

    # Read the training parquet
    df = pd.read_parquet(TRAINING_DATA_URI)
    logger.info("Loaded training data: %d rows, %d columns", len(df), len(df.columns))

    # Keep only the features that matter for monitoring
    available = [f for f in MONITORED_FEATURES if f in df.columns]
    missing = [f for f in MONITORED_FEATURES if f not in df.columns]
    if missing:
        logger.warning("Features not in training data (skipped): %s", missing)

    baseline_df = df[available].dropna()
    logger.info(
        "Baseline subset: %d rows, %d features: %s",
        len(baseline_df),
        len(available),
        available,
    )

    # Write CSV to a temp file, upload to GCS
    csv_path = "/tmp/training_baseline.csv"
    baseline_df.to_csv(csv_path, index=False)

    # Upload to GCS
    dest = MONITORING_DATASET_URI.replace("gs://", "")
    bucket_name = dest.split("/")[0]
    blob_path = "/".join(dest.split("/")[1:])

    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).upload_from_filename(csv_path)
    logger.info("Uploaded baseline CSV to %s (%d rows)", MONITORING_DATASET_URI, len(baseline_df))

    # Upload schema YAML files alongside the baseline
    for schema_name, gcs_uri in [
        ("predict_instance_schema.yaml", PREDICT_SCHEMA_URI),
        ("analysis_instance_schema.yaml", ANALYSIS_SCHEMA_URI),
    ]:
        local_schema = os.path.join(_SCHEMA_DIR, schema_name)
        if not os.path.exists(local_schema):
            logger.warning("Schema file not found: %s", local_schema)
            continue
        dest = gcs_uri.replace("gs://", "")
        b_name = dest.split("/")[0]
        b_path = "/".join(dest.split("/")[1:])
        client.bucket(b_name).blob(b_path).upload_from_filename(local_schema)
        logger.info("Uploaded schema %s to %s", schema_name, gcs_uri)

    return MONITORING_DATASET_URI


def _create_monitoring_job(
    endpoint: aiplatform.Endpoint,
    baseline_uri: str,
    dry_run: bool = False,
) -> None:
    """Create a Vertex AI Model Monitoring job on the endpoint.

    Configures:
      - Prediction drift detection (compare recent predictions to baseline)
      - Feature skew detection (compare serving inputs to training distribution)
      - Logging to BigQuery for all requests/responses
    """
    if dry_run:
        logger.info(
            "[DRY RUN] Would create Model Monitoring job on endpoint %s "
            "with baseline %s",
            endpoint.resource_name,
            baseline_uri,
        )
        return

    # Build drift detection config — threshold per feature
    drift_thresholds = {
        feat: DEFAULT_DRIFT_THRESHOLD for feat in MONITORED_FEATURES
    }
    skew_thresholds = {
        feat: DEFAULT_DRIFT_THRESHOLD for feat in MONITORED_FEATURES
    }

    # Objective config: skew + drift
    skew_config = aiplatform.model_monitoring.SkewDetectionConfig(
        data_source=baseline_uri,
        skew_thresholds=skew_thresholds,
        target_field="service_headway",
        data_format="csv",
    )
    drift_config = aiplatform.model_monitoring.DriftDetectionConfig(
        drift_thresholds=drift_thresholds,
    )
    objective_config = aiplatform.model_monitoring.ObjectiveConfig(
        skew_detection_config=skew_config,
        drift_detection_config=drift_config,
    )

    # Sampling — log 100% of requests initially (low traffic; reduce later)
    random_sampling = aiplatform.model_monitoring.RandomSampleConfig(
        sample_rate=1.0,
    )

    # Alert — email notification on threshold breach
    alert_config = aiplatform.model_monitoring.EmailAlertConfig(
        user_emails=["dan.herman@me.com"],
    )

    # Schedule — run monitoring analysis every hour during initial deployment.
    # Once baseline JS divergence scores stabilize (~1 week), consider widening
    # to 6h or 24h via the Vertex AI console or by re-running this script.
    schedule_config = aiplatform.model_monitoring.ScheduleConfig(
        monitor_interval=1,  # hours
    )

    logger.info("Creating Model Monitoring job...")
    job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name=f"{MODEL_DISPLAY_NAME}-monitoring",
        endpoint=endpoint,
        logging_sampling_strategy=random_sampling,
        schedule_config=schedule_config,
        alert_config=alert_config,
        objective_configs=objective_config,
        predict_instance_schema_uri=PREDICT_SCHEMA_URI,
        analysis_instance_schema_uri=ANALYSIS_SCHEMA_URI,
        project=PROJECT_ID,
        location=LOCATION,
    )
    logger.info("Model Monitoring job created: %s", job.resource_name)
    logger.info(
        "Monitoring will check for drift every 1 hour. "
        "Thresholds: JS divergence > %.2f per feature.",
        DEFAULT_DRIFT_THRESHOLD,
    )


def _smoke_test(endpoint: aiplatform.Endpoint) -> None:
    """Send a synthetic prediction request to verify the endpoint is alive."""
    logger.info("Running smoke test...")

    # Build a fake 20-observation window
    fake_obs = {
        "service_headway": 5.0,
        "preceding_train_gap": 4.5,
        "upstream_headway_14th": 5.2,
        "travel_time_14th": 2.1,
        "travel_time_14th_deviation": 0.05,
        "travel_time_23rd": 1.8,
        "travel_time_23rd_deviation": 0.03,
        "travel_time_34th": 2.3,
        "travel_time_34th_deviation": 0.04,
        "stops_at_23rd": 1.0,
        "hour_sin": 0.87,
        "hour_cos": -0.50,
        "time_idx": 14200,
        "empirical_median": 5.0,
        "day_of_week": 2,
        "route_id": "A",
        "regime_id": "Day",
        "track_id": "A1",
        "preceding_route_id": "A",
    }
    instance = {
        "group_id": "A_South",
        "observations": [fake_obs] * 20,
    }

    start = time.time()
    response = endpoint.predict(instances=[instance])
    elapsed_ms = (time.time() - start) * 1000

    pred = response.predictions[0]
    logger.info(
        "Smoke test passed ✓  P10=%.2f  P50=%.2f  P90=%.2f  (%.0f ms)",
        pred["headway_p10"],
        pred["headway_p50"],
        pred["headway_p90"],
        elapsed_ms,
    )


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Deploy headway-tft model to Vertex AI Endpoint with monitoring"
    )
    parser.add_argument(
        "--skip-monitoring",
        action="store_true",
        help="Deploy the model but skip Model Monitoring setup",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip the synthetic prediction smoke test",
    )
    parser.add_argument(
        "--monitoring-only",
        action="store_true",
        help="Skip deployment — only set up monitoring + smoke test on existing endpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making any changes",
    )
    args = parser.parse_args()

    # Initialize Vertex AI SDK
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    if not args.monitoring_only:
        # Step 1: Find the latest registered model
        logger.info("=" * 60)
        logger.info("Step 1: Finding latest registered model")
        logger.info("=" * 60)
        model = _find_latest_model()

        # Step 2: Get or create the endpoint
        logger.info("=" * 60)
        logger.info("Step 2: Getting or creating endpoint")
        logger.info("=" * 60)
        endpoint = _get_or_create_endpoint()

        # Step 3: Deploy model to endpoint
        logger.info("=" * 60)
        logger.info("Step 3: Deploying model to endpoint")
        logger.info("=" * 60)
        _deploy_model(endpoint, model, dry_run=args.dry_run)
    else:
        logger.info("=" * 60)
        logger.info("--monitoring-only: skipping deploy, resolving existing endpoint")
        logger.info("=" * 60)
        model = _find_latest_model()
        endpoint = _get_or_create_endpoint()

    # Step 4: Model Monitoring
    if not args.skip_monitoring:
        logger.info("=" * 60)
        logger.info("Step 4a: Generating monitoring baseline")
        logger.info("=" * 60)
        baseline_uri = _generate_monitoring_baseline(dry_run=args.dry_run)

        logger.info("=" * 60)
        logger.info("Step 4b: Creating Model Monitoring job")
        logger.info("=" * 60)
        _create_monitoring_job(endpoint, baseline_uri, dry_run=args.dry_run)
    else:
        logger.info("Skipping Model Monitoring (--skip-monitoring)")

    # Step 5: Smoke test
    if not args.skip_smoke_test and not args.dry_run:
        logger.info("=" * 60)
        logger.info("Step 5: Smoke test")
        logger.info("=" * 60)
        _smoke_test(endpoint)
    else:
        logger.info("Skipping smoke test")

    # Summary
    logger.info("=" * 60)
    logger.info("DEPLOYMENT COMPLETE")
    logger.info("=" * 60)
    logger.info("  Model:    %s", model.resource_name)
    logger.info("  Endpoint: %s", endpoint.resource_name)
    logger.info(
        "  Endpoint URL: https://%s-aiplatform.googleapis.com/v1/%s:predict",
        LOCATION,
        endpoint.resource_name,
    )
    if not args.skip_monitoring:
        logger.info("  Monitoring: ENABLED (1h interval, JS threshold %.2f)",
                     DEFAULT_DRIFT_THRESHOLD)
    if args.monitoring_only:
        logger.info("  Mode: monitoring-only (deployment skipped)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
