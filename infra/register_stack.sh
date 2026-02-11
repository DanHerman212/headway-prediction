#!/bin/bash
# =============================================================================
# register_stack.sh
#
# Runs on your LOCAL machine after both servers are deployed.
# Connects the ZenML client to the remote server and registers:
#   - GCS artifact store
#   - Vertex AI experiment tracker (writes to Vertex AI TensorBoard)
#   - Combined production stack
#
# Usage: ./infra/register_stack.sh
# =============================================================================
set -euo pipefail

# Configuration — must match setup_mlops_infra.sh
PROJECT_ID="realtime-headway-prediction"
REGION="us-east1"
ARTIFACT_BUCKET="mlops-artifacts-${PROJECT_ID}"
AR_REPO_NAME="mlops-images"
AR_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}"

# Vertex AI TensorBoard instance ID created via:
#   gcloud ai tensorboards create --display-name="headway-training" ...
TENSORBOARD_ID="8313539359009669120"

# Get deployed service URLs
echo "--- Retrieving Cloud Run service URLs ---"
ZENML_URL=$(gcloud run services describe zenml-server --region "${REGION}" --format="value(status.url)")

echo "  ZenML Server:  ${ZENML_URL}"
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
# Step 3: Register Vertex AI experiment tracker
# -------------------------------------------------
echo "--- Step 3: Registering Vertex AI experiment tracker ---"
if zenml experiment-tracker describe vertex_tracker &>/dev/null; then
  echo "  ✓ vertex_tracker already registered"
else
  zenml experiment-tracker register vertex_tracker \
    --flavor=vertex \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --experiment_tensorboard="${TENSORBOARD_ID}" \
    --staging_bucket="gs://${ARTIFACT_BUCKET}"
  echo "  ✓ vertex_tracker registered"
fi
echo ""

# -------------------------------------------------
# Step 4: Register Container Registry & Image Builder
# -------------------------------------------------
echo "--- Step 4: Registering Container Registry & Image Builder ---"
# Container Registry (Artifact Registry)
if zenml container-registry describe gcp_registry &>/dev/null; then
  echo "  ✓ gcp_registry already registered"
else
  zenml container-registry register gcp_registry \
    --flavor=gcp \
    --uri="${AR_URI}"
  echo "  ✓ gcp_registry registered"
fi

# Image Builder (Cloud Build) - Required since we can't build locally
if zenml image-builder describe gcp_image_builder &>/dev/null; then
  echo "  ✓ gcp_image_builder already registered"
else
  zenml image-builder register gcp_image_builder \
    --flavor=gcp
  echo "  ✓ gcp_image_builder registered"
fi
echo ""

# -------------------------------------------------
# Step 5: Register Vertex AI Orchestrator
# -------------------------------------------------
echo "--- Step 5: Registering Vertex AI Orchestrator ---"
if zenml orchestrator describe vertex_orchestrator &>/dev/null; then
  echo "  ✓ vertex_orchestrator already registered"
else
  zenml orchestrator register vertex_orchestrator \
    --flavor=vertex \
    --location="${REGION}" \
    --workload_service_account="${PROJECT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
    # Note: Using default compute SA or specific MLOps SA if you have one formatted differently
    # Based on setup_mlops_infra.sh, the SA is mlops-sa@...
    # Let's try to infer it or default to None (ZenML will confirm)
  echo "  ✓ vertex_orchestrator registered"
fi
echo ""

# -------------------------------------------------
# Step 6: Create and activate the stack
# -------------------------------------------------
echo "--- Step 6: Registering production stack ---"
if zenml stack describe gcp_production_stack &>/dev/null; then
  echo "  ✓ gcp_production_stack already registered"
else
  zenml stack register gcp_production_stack \
    -o vertex_orchestrator \
    -a gcs_store \
    -e vertex_tracker \
    -c gcp_registry \
    -i gcp_image_builder
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
echo " Next: Install requirements:"
echo "       zenml integration install gcp"
echo ""
