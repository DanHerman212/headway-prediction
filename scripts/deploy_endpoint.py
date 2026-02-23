#!/usr/bin/env python3
"""
deploy_endpoint.py
------------------
Deploy the latest headway-tft model to a Vertex AI endpoint with monitoring.

Usage:
  python scripts/deploy_endpoint.py
  python scripts/deploy_endpoint.py --refresh-image
  python scripts/deploy_endpoint.py --skip-monitoring
"""

import argparse
import logging
import os
import sys

import pandas as pd
from google.cloud import aiplatform
from google.cloud import storage as gcs_storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -- Config --
PROJECT_ID = "realtime-headway-prediction"
LOCATION = "us-east1"
MODEL_DISPLAY_NAME = "headway-tft"
ENDPOINT_DISPLAY_NAME = "headway-prediction-endpoint"

ARTIFACT_BUCKET = "gs://mlops-artifacts-realtime-headway-prediction"
TRAINING_DATA_URI = f"{ARTIFACT_BUCKET}/data/training_data.parquet"
MONITORING_BASELINE_URI = f"{ARTIFACT_BUCKET}/monitoring/training_baseline.csv"
PREDICT_SCHEMA_URI = f"{ARTIFACT_BUCKET}/monitoring/predict_instance_schema.yaml"
ANALYSIS_SCHEMA_URI = f"{ARTIFACT_BUCKET}/monitoring/analysis_instance_schema.yaml"

SERVING_CONTAINER_URI = (
    "us-east1-docker.pkg.dev/realtime-headway-prediction/"
    "mlops-images/headway-serving-cpr:latest"
)

MACHINE_TYPE = "n1-standard-4"
SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "..", "infra", "schemas")

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

DRIFT_THRESHOLD = 0.3


def find_model(refresh_image=False):
    """Find the latest headway-tft model. Optionally re-register with fresh image."""
    models = aiplatform.Model.list(
        filter=f'display_name="{MODEL_DISPLAY_NAME}"',
        order_by="create_time desc",
    )
    if not models:
        logger.error("No model found with display_name='%s'.", MODEL_DISPLAY_NAME)
        sys.exit(1)

    model = models[0]
    logger.info("Found model: %s (created %s)", model.resource_name, model.create_time)

    if not refresh_image:
        return model

    logger.info("Re-registering model with fresh container image...")
    new_model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=model.gca_resource.artifact_uri,
        serving_container_image_uri=SERVING_CONTAINER_URI,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        labels=dict(model.labels) if model.labels else {},
        description=(model.gca_resource.description or "").replace(" [image-refresh]", "")
        + " [image-refresh]",
    )
    logger.info("New model: %s", new_model.resource_name)
    try:
        model.delete()
        logger.info("Deleted old model: %s", model.resource_name)
    except Exception as e:
        logger.warning("Could not delete old model: %s", e)
    return new_model


def create_endpoint():
    """Create a new endpoint. Exits if one already exists."""
    existing = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"',
    )
    if existing:
        eid = existing[0].resource_name.split("/")[-1]
        logger.error(
            "Endpoint '%s' already exists. Delete it first:\n"
            "  gcloud ai endpoints delete %s --region=%s --quiet",
            ENDPOINT_DISPLAY_NAME, eid, LOCATION,
        )
        sys.exit(1)

    logger.info("Creating endpoint: %s", ENDPOINT_DISPLAY_NAME)
    endpoint = aiplatform.Endpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        project=PROJECT_ID,
        location=LOCATION,
    )
    logger.info("Created: %s", endpoint.resource_name)
    return endpoint


def deploy_model(endpoint, model):
    """Deploy model to endpoint. Blocks until ready (~15-20 min)."""
    logger.info("Deploying model (machine_type=%s)... ~15-20 minutes.", MACHINE_TYPE)
    endpoint.deploy(
        model=model,
        deployed_model_display_name=f"{MODEL_DISPLAY_NAME}-serving",
        machine_type=MACHINE_TYPE,
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
        enable_access_logging=True,
        deploy_request_timeout=1800,
    )
    logger.info("Model deployed successfully.")


def upload_monitoring_baseline():
    """Generate and upload a training feature baseline CSV for monitoring."""
    logger.info("Generating monitoring baseline from %s", TRAINING_DATA_URI)
    df = pd.read_parquet(TRAINING_DATA_URI)
    available = [f for f in MONITORED_FEATURES if f in df.columns]
    baseline_df = df[available].dropna()
    logger.info("Baseline: %d rows, %d features", len(baseline_df), len(available))

    csv_path = "/tmp/training_baseline.csv"
    baseline_df.to_csv(csv_path, index=False)

    dest = MONITORING_BASELINE_URI.replace("gs://", "")
    bucket_name, blob_path = dest.split("/", 1)
    client = gcs_storage.Client()
    client.bucket(bucket_name).blob(blob_path).upload_from_filename(csv_path)
    logger.info("Uploaded baseline to %s", MONITORING_BASELINE_URI)

    for name, uri in [
        ("predict_instance_schema.yaml", PREDICT_SCHEMA_URI),
        ("analysis_instance_schema.yaml", ANALYSIS_SCHEMA_URI),
    ]:
        local = os.path.join(SCHEMA_DIR, name)
        if os.path.exists(local):
            d = uri.replace("gs://", "")
            b, p = d.split("/", 1)
            client.bucket(b).blob(p).upload_from_filename(local)
            logger.info("Uploaded %s", name)

    return MONITORING_BASELINE_URI


def create_monitoring_job(endpoint, baseline_uri):
    """Create a Model Monitoring job on the endpoint."""
    thresholds = {f: DRIFT_THRESHOLD for f in MONITORED_FEATURES}

    # Check for existing active job on this endpoint
    try:
        all_jobs = aiplatform.ModelDeploymentMonitoringJob.list(
            project=PROJECT_ID, location=LOCATION,
        )
        active = [
            j for j in all_jobs
            if getattr(j, "endpoint", None)
            and endpoint.resource_name in j.endpoint
            and j.state.name not in (
                "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"
            )
        ]
        if active:
            logger.info("Monitoring job already exists: %s", active[0].resource_name)
            return
    except Exception as e:
        logger.warning("Could not check existing monitoring jobs: %s", e)

    logger.info("Creating monitoring job...")
    job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name=f"{MODEL_DISPLAY_NAME}-monitoring",
        endpoint=endpoint,
        logging_sampling_strategy=aiplatform.model_monitoring.RandomSampleConfig(
            sample_rate=1.0,
        ),
        schedule_config=aiplatform.model_monitoring.ScheduleConfig(
            monitor_interval=1,
        ),
        alert_config=aiplatform.model_monitoring.EmailAlertConfig(
            user_emails=["dan.herman@me.com"],
        ),
        objective_configs=aiplatform.model_monitoring.ObjectiveConfig(
            skew_detection_config=aiplatform.model_monitoring.SkewDetectionConfig(
                data_source=baseline_uri,
                skew_thresholds=thresholds,
                target_field="service_headway",
                data_format="csv",
            ),
            drift_detection_config=aiplatform.model_monitoring.DriftDetectionConfig(
                drift_thresholds=thresholds,
            ),
        ),
        predict_instance_schema_uri=PREDICT_SCHEMA_URI,
        analysis_instance_schema_uri=ANALYSIS_SCHEMA_URI,
        project=PROJECT_ID,
        location=LOCATION,
    )
    logger.info("Monitoring job created: %s", job.resource_name)


def main():
    parser = argparse.ArgumentParser(
        description="Deploy headway-tft to Vertex AI with monitoring",
    )
    parser.add_argument(
        "--refresh-image", action="store_true",
        help="Re-register model to pick up a rebuilt container image",
    )
    parser.add_argument(
        "--skip-monitoring", action="store_true",
        help="Deploy only, skip monitoring setup",
    )
    args = parser.parse_args()

    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # 1. Find model
    logger.info("=" * 60)
    logger.info("Step 1: Finding model")
    logger.info("=" * 60)
    model = find_model(refresh_image=args.refresh_image)

    # 2. Create endpoint
    logger.info("=" * 60)
    logger.info("Step 2: Creating endpoint")
    logger.info("=" * 60)
    endpoint = create_endpoint()

    # 3. Deploy
    logger.info("=" * 60)
    logger.info("Step 3: Deploying model")
    logger.info("=" * 60)
    deploy_model(endpoint, model)

    # 4. Monitoring
    if not args.skip_monitoring:
        logger.info("=" * 60)
        logger.info("Step 4: Setting up monitoring")
        logger.info("=" * 60)
        baseline_uri = upload_monitoring_baseline()
        create_monitoring_job(endpoint, baseline_uri)

    # Done
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("  Endpoint: %s", endpoint.resource_name)
    logger.info("  Monitoring: %s", "ENABLED" if not args.skip_monitoring else "SKIPPED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
