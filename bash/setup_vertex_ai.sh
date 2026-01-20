#!/bin/bash
#
# Setup Vertex AI Environment
#
# This script:
#   1. Enables required GCP APIs
#   2. Creates Artifact Registry repository
#   3. Grants permissions to Vertex AI service agent
#
# Run this once before deploying pipelines.
#
# Usage:
#   ./setup_vertex_ai.sh

set -e

# Configuration
PROJECT_ID="realtime-headway-prediction"
REGION="us-east1"
REPOSITORY="ml-pipelines"

echo "=========================================="
echo "Vertex AI Environment Setup"
echo "=========================================="
echo ""
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Repository: ${REPOSITORY}"
echo ""

# Step 1: Enable required APIs
echo "=========================================="
echo "Step 1: Enabling Required APIs"
echo "=========================================="

REQUIRED_APIS=(
    "aiplatform.googleapis.com"
    "artifactregistry.googleapis.com"
    "cloudbuild.googleapis.com"
    "storage.googleapis.com"
    "bigquery.googleapis.com"
)

for API in "${REQUIRED_APIS[@]}"; do
    if gcloud services list --enabled --project=${PROJECT_ID} --filter="config.name=${API}" --format="value(config.name)" | grep -q "${API}"; then
        echo "✓ ${API} is enabled"
    else
        echo "  Enabling ${API}..."
        gcloud services enable ${API} --project=${PROJECT_ID}
        echo "✓ ${API} enabled"
    fi
done
echo ""

# Step 2: Create Artifact Registry repository if it doesn't exist
echo "=========================================="
echo "Step 2: Creating Artifact Registry"
echo "=========================================="
if ! gcloud artifacts repositories describe ${REPOSITORY} \
    --location=${REGION} \
    --project=${PROJECT_ID} &>/dev/null; then
    
    echo "Creating repository: ${REPOSITORY}"
    gcloud artifacts repositories create ${REPOSITORY} \
        --repository-format=docker \
        --location=${REGION} \
        --project=${PROJECT_ID} \
        --description="ML pipelines for headway prediction"
    echo "✓ Repository created successfully"
else
    echo "✓ Repository ${REPOSITORY} already exists"
fi
echo ""

# Step 3: Grant permissions to Vertex AI service agent
echo "=========================================="
echo "Step 3: Configuring Permissions"
echo "=========================================="

# Get the project number
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
VERTEX_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "Vertex AI Service Agent: ${VERTEX_SA}"
echo "Compute Service Account: ${COMPUTE_SA}"
echo ""

# Permissions to check/grant for Vertex AI service agent (role:description pairs)
VERTEX_PERMISSIONS=(
    "roles/artifactregistry.reader:Artifact Registry Reader"
    "roles/storage.objectAdmin:Storage Object Admin"
    "roles/bigquery.dataEditor:BigQuery Data Editor"
    "roles/bigquery.jobUser:BigQuery Job User"
    "roles/aiplatform.user:Vertex AI User"
)

# Check and grant permissions for Vertex AI service agent
for PERMISSION in "${VERTEX_PERMISSIONS[@]}"; do
    ROLE="${PERMISSION%%:*}"
    ROLE_NAME="${PERMISSION#*:}"
    
    if gcloud projects get-iam-policy ${PROJECT_ID} \
        --flatten="bindings[].members" \
        --filter="bindings.members:serviceAccount:${VERTEX_SA} AND bindings.role:${ROLE}" \
        --format="value(bindings.role)" 2>/dev/null | grep -q "${ROLE}"; then
        echo "✓ ${ROLE_NAME} already configured"
    else
        echo "  Granting ${ROLE_NAME}..."
        gcloud projects add-iam-policy-binding ${PROJECT_ID} \
            --member="serviceAccount:${VERTEX_SA}" \
            --role="${ROLE}" \
            --quiet
        echo "✓ ${ROLE_NAME} granted"
    fi
done

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "You can now deploy pipelines using:"
echo "  cd ml_pipelines/A1"
echo "  ./deploy.sh baseline1"
echo ""
