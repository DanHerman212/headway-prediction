#!/bin/bash
# =============================================================================
# register_stack.sh
#
# Runs on your LOCAL machine after both servers are deployed.
# Connects the ZenML client to the remote server and registers:
#   - GCS artifact store
#   - Remote MLflow experiment tracker
#   - Combined production stack
#
# Usage: ./infra/register_stack.sh
# =============================================================================
set -euo pipefail

# Configuration — must match setup_mlops_infra.sh
PROJECT_ID="realtime-headway-prediction"
REGION="us-east1"
ARTIFACT_BUCKET="mlops-artifacts-${PROJECT_ID}"

# Get deployed service URLs
echo "--- Retrieving Cloud Run service URLs ---"
ZENML_URL=$(gcloud run services describe zenml-server --region "${REGION}" --format="value(status.url)")
MLFLOW_URL=$(gcloud run services describe mlflow-server --region "${REGION}" --format="value(status.url)")

echo "  ZenML Server:  ${ZENML_URL}"
echo "  MLflow Server: ${MLFLOW_URL}"
echo ""

# -------------------------------------------------
# Step 1: Connect local ZenML client to remote server
# -------------------------------------------------
echo "--- Step 1: Connecting to remote ZenML server ---"
echo "  (Default credentials: admin / empty password)"
zenml connect --url "${ZENML_URL}" --no-verify-ssl
echo ""

# -------------------------------------------------
# Step 2: Register artifact store (GCS)
# -------------------------------------------------
echo "--- Step 2: Registering GCS artifact store ---"
if zenml artifact-store describe gcs_store &>/dev/null; then
  echo "  ✓ gcs_store already registered"
else
  zenml artifact-store register gcs_store \
    --flavor=gcp \
    --path="gs://${ARTIFACT_BUCKET}/zenml"
  echo "  ✓ gcs_store registered"
fi
echo ""

# -------------------------------------------------
# Step 3: Register MLflow experiment tracker
# -------------------------------------------------
echo "--- Step 3: Registering MLflow experiment tracker ---"
if zenml experiment-tracker describe mlflow_tracker &>/dev/null; then
  echo "  ✓ mlflow_tracker already registered"
else
  zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri="${MLFLOW_URL}"
  echo "  ✓ mlflow_tracker registered"
fi
echo ""

# -------------------------------------------------
# Step 4: Create and activate the stack
# -------------------------------------------------
echo "--- Step 4: Registering production stack ---"
if zenml stack describe gcp_production_stack &>/dev/null; then
  echo "  ✓ gcp_production_stack already registered"
else
  zenml stack register gcp_production_stack \
    -o default \
    -a gcs_store \
    -e mlflow_tracker
  echo "  ✓ gcp_production_stack registered"
fi

zenml stack set gcp_production_stack
echo ""

echo "=========================================="
echo " ✅ Stack Registration Complete!"
echo "=========================================="
echo ""
zenml stack describe
echo ""
echo " Next: python infra/verify_deployment.py"
echo ""
