#!/bin/bash
# =============================================================================
# deploy_mlflow.sh
#
# Builds the MLflow Docker image and deploys it to Cloud Run.
# Prerequisites: Run setup_mlops_infra.sh first.
#
# Usage: ./infra/deploy_mlflow.sh
# =============================================================================
set -euo pipefail

# Configuration — must match setup_mlops_infra.sh
PROJECT_ID="realtime-headway-prediction"
REGION="us-east1"
AR_REPO="mlops-images"
SQL_INSTANCE="mlops-metadata"
SERVICE_ACCOUNT="mlops-sa"
SECRET_NAME="mlops-db-pass"
MLFLOW_DB="mlflow_db"
DB_USER="mlops-user"
ARTIFACT_BUCKET="mlops-artifacts-${PROJECT_ID}"

# Derived
SA_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/mlflow-server:v1"
INSTANCE_CONNECTION_NAME=$(gcloud sql instances describe "${SQL_INSTANCE}" --format="value(connectionName)")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo " Deploying MLflow Tracking Server"
echo "=========================================="
echo " Image: ${IMAGE}"
echo " SQL:   ${INSTANCE_CONNECTION_NAME}"
echo ""

# -------------------------------------------------
# Step 1: Build & Push via Cloud Build
# -------------------------------------------------
echo "--- Step 1: Building image via Cloud Build ---"
gcloud builds submit "${SCRIPT_DIR}" \
  --config "${SCRIPT_DIR}/cloudbuild_mlflow.yaml" \
  --substitutions "_IMAGE_TAG=${IMAGE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --quiet
echo "✓ Image built and pushed to Artifact Registry"
echo ""

# -------------------------------------------------
# Step 2: Deploy to Cloud Run
# -------------------------------------------------
echo "--- Step 2: Deploying to Cloud Run ---"
gcloud run deploy mlflow-server \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --service-account "${SA_EMAIL}" \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 300 \
  --add-cloudsql-instances "${INSTANCE_CONNECTION_NAME}" \
  --set-secrets "DB_PASSWORD=${SECRET_NAME}:latest" \
  --set-env-vars "\
INSTANCE_CONNECTION_NAME=${INSTANCE_CONNECTION_NAME},\
DB_USER=${DB_USER},\
MLFLOW_DB=${MLFLOW_DB},\
ARTIFACT_BUCKET=${ARTIFACT_BUCKET}" \
  --quiet

MLFLOW_URL=$(gcloud run services describe mlflow-server --region "${REGION}" --format="value(status.url)")
echo ""
echo "=========================================="
echo " ✅ MLflow Server Deployed!"
echo "=========================================="
echo " URL: ${MLFLOW_URL}"
echo ""
echo " Next: ./infra/deploy_zenml.sh"
echo ""
