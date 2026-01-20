#!/bin/bash
#
# Deploy A1 Training Pipeline to Vertex AI
#
# This script:
#   1. Builds container image using Cloud Build
#   2. Compiles Kubeflow pipeline
#   3. Submits pipeline run to Vertex AI
#
# Usage:
#   ./deploy.sh [RUN_NAME] [--skip-build]
#
# Example:
#   ./deploy.sh baseline_001
#   ./deploy.sh baseline_002 --skip-build   # Skip container rebuild

set -e

# Configuration
PROJECT_ID="realtime-headway-prediction"
REGION="us-east1"
REPOSITORY="ml-pipelines"
IMAGE_NAME="a1-training"

# Get run name from argument or generate timestamp-based name
if [ -z "$1" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="run_${TIMESTAMP}"
else
    RUN_NAME="$1"
fi

echo "=========================================="
echo "A1 Pipeline Deployment"
echo "=========================================="
echo ""
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Run Name: ${RUN_NAME}"
echo ""

# Step 0: Enable required APIs
echo "=========================================="
echo "Step 0: Checking Required APIs"
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

# Step 1: Create Artifact Registry repository if it doesn't exist
echo "=========================================="
echo "Step 1: Checking Artifact Registry"
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
    echo "Repository created successfully"
else
    echo "Repository ${REPOSITORY} already exists"
fi
echo ""

# Step 1.5: Grant permissions to Vertex AI service agent
echo "=========================================="
echo "Step 1.5: Checking Permissions"
echo "=========================================="

# Get the project number
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
VERTEX_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "Vertex AI Service Agent: ${VERTEX_SA}"
echo "Compute Service Account: ${COMPUTE_SA}"
echo ""

# Array of permissions to check/grant for Vertex AI service agent
declare -A VERTEX_PERMISSIONS
VERTEX_PERMISSIONS["roles/artifactregistry.reader"]="Artifact Registry Reader"
VERTEX_PERMISSIONS["roles/storage.objectAdmin"]="Storage Object Admin"
VERTEX_PERMISSIONS["roles/bigquery.dataEditor"]="BigQuery Data Editor"
VERTEX_PERMISSIONS["roles/bigquery.jobUser"]="BigQuery Job User"
VERTEX_PERMISSIONS["roles/aiplatform.user"]="Vertex AI User"

# Check and grant permissions for Vertex AI service agent
for ROLE in "${!VERTEX_PERMISSIONS[@]}"; do
    ROLE_NAME="${VERTEX_PERMISSIONS[$ROLE]}"
    
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
echo "Permissions check complete"
echo ""

# Define image URI (needed for checks)
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"

# Step 2: Build container using Cloud Build
echo "=========================================="
echo "Step 2: Building Container (Cloud Build)"
echo "=========================================="

# Check if we should skip build
SKIP_BUILD=false
if [ "$2" == "--skip-build" ]; then
    SKIP_BUILD=true
    echo "Skipping container build (--skip-build flag)"
elif gcloud artifacts docker images describe ${IMAGE_URI} --project=${PROJECT_ID} &>/dev/null; then
    echo "Container image already exists: ${IMAGE_URI}"
    read -p "Skip rebuild? (y/n) [default: n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SKIP_BUILD=true
        echo "Skipping container build"
    fi
fi

if [ "$SKIP_BUILD" = false ]; then
    echo "Submitting build to Cloud Build..."
    echo ""
    
    gcloud builds submit \
        --config=cloudbuild.yaml \
        --project=${PROJECT_ID} \
        --region=${REGION} \
        .
    
    echo ""
    echo "Container build complete!"
    echo "Image: ${IMAGE_URI}"
else
    echo "Using existing image: ${IMAGE_URI}"
fi
echo ""

# Step 3: Compile pipeline
echo "=========================================="
echo "Step 3: Compiling Pipeline"
echo "=========================================="
echo "Compiling Kubeflow pipeline..."
python3 pipeline.py --compile
echo "Pipeline compiled to: a1_pipeline.yaml"
echo ""

# Step 4: Submit pipeline to Vertex AI
echo "=========================================="
echo "Step 4: Submitting to Vertex AI Pipelines"
echo "=========================================="
echo "Submitting pipeline run: ${RUN_NAME}"
echo ""

python3 pipeline.py --submit --run_name ${RUN_NAME}

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Run Name: ${RUN_NAME}"
echo "Image: ${IMAGE_URI}"
echo ""
echo "Monitor your pipeline:"
echo "  Pipelines: https://console.cloud.google.com/vertex-ai/pipelines?project=${PROJECT_ID}"
echo "  Experiments: https://console.cloud.google.com/vertex-ai/experiments?project=${PROJECT_ID}"
echo ""
echo "View TensorBoard:"
echo "  gs://ml-pipelines-headway-prediction/tensorboard/A1/${RUN_NAME}"
echo ""
