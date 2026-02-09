#!/bin/bash
# =============================================================================
# deploy_zenml.sh
#
# Builds the ZenML Server Docker image and deploys it to Cloud Run.
# Prerequisites: Run setup_mlops_infra.sh first.
#
# Usage: ./infra/deploy_zenml.sh
# =============================================================================
set -euo pipefail

# Configuration — must match setup_mlops_infra.sh
PROJECT_ID="realtime-headway-prediction"
REGION="us-east1"
AR_REPO="mlops-images"
SQL_INSTANCE="mlops-metadata"
SERVICE_ACCOUNT="mlops-sa"
SECRET_NAME="mlops-db-pass"
ZENML_DB="zenml_db"
DB_USER="mlops-user"

# Derived
SA_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/zenml-server:v1"
INSTANCE_CONNECTION_NAME=$(gcloud sql instances describe "${SQL_INSTANCE}" --format="value(connectionName)")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo " Deploying ZenML Server"
echo "=========================================="
echo " Image: ${IMAGE}"
echo " SQL:   ${INSTANCE_CONNECTION_NAME}"
echo ""

# -------------------------------------------------
# Step 1: Build & Push via Cloud Build
# -------------------------------------------------
echo "--- Step 1: Building image via Cloud Build ---"
gcloud builds submit "${SCRIPT_DIR}" \
  --config "${SCRIPT_DIR}/cloudbuild_zenml.yaml" \
  --substitutions "_IMAGE_TAG=${IMAGE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --quiet
echo "✓ Image built and pushed to Artifact Registry"
echo ""

# -------------------------------------------------
# Step 2: Deploy to Cloud Run (multi-container with sidecar)
# -------------------------------------------------
echo "--- Step 2: Deploying to Cloud Run ---"
# ZenML's URL validator rejects ?unix_socket=... query parameters, so the
# built-in --add-cloudsql-instances flag (Unix socket only) cannot be used.
# Instead, we deploy a Cloud SQL Auth Proxy sidecar in TCP mode via a
# service YAML. The proxy listens on 127.0.0.1:3306 inside the pod.

# Generate the service YAML from the template
YAML_TEMPLATE="${SCRIPT_DIR}/zenml-cloudrun-service.yaml"
YAML_RENDERED="/tmp/zenml-cloudrun-service-rendered.yaml"

sed \
  -e "s|__IMAGE__|${IMAGE}|g" \
  -e "s|__SA_EMAIL__|${SA_EMAIL}|g" \
  -e "s|__INSTANCE_CONNECTION_NAME__|${INSTANCE_CONNECTION_NAME}|g" \
  -e "s|__DB_USER__|${DB_USER}|g" \
  -e "s|__ZENML_DB__|${ZENML_DB}|g" \
  -e "s|__SECRET_NAME__|${SECRET_NAME}|g" \
  -e "s|__REGION__|${REGION}|g" \
  "${YAML_TEMPLATE}" > "${YAML_RENDERED}"

echo "  Generated service YAML → ${YAML_RENDERED}"

# Deploy the multi-container service
gcloud run services replace "${YAML_RENDERED}" \
  --region "${REGION}" \
  --quiet

# Allow unauthenticated access (gcloud run services replace doesn't set IAM)
gcloud run services add-iam-policy-binding zenml-server \
  --region "${REGION}" \
  --member="allUsers" \
  --role="roles/run.invoker" \
  --quiet > /dev/null 2>&1

rm -f "${YAML_RENDERED}"

ZENML_URL=$(gcloud run services describe zenml-server --region "${REGION}" --format="value(status.url)")
echo ""
echo "=========================================="
echo " ✅ ZenML Server Deployed!"
echo "=========================================="
echo " URL: ${ZENML_URL}"
echo " Default credentials: admin / (empty password on first login)"
echo ""
echo " Next: ./infra/register_stack.sh"
echo ""
