#!/bin/bash
set -e

# ==============================================================================
# Configuration
# ==============================================================================

# Parse command line arguments
SKIP_BUILD=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-build) SKIP_BUILD=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Load .env file if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")" # Points to ml_pipelines root

if [ -f "${PROJECT_ROOT}/ml_pipelines/.env" ]; then
    echo "Loading configuration from .env..."
    set -a
    source "${PROJECT_ROOT}/ml_pipelines/.env"
    set +a
fi

PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
REGION=${VERTEX_LOCATION:-"us-east1"}
ARTIFACT_REGION=${ARTIFACT_REGION:-"us-east1"}
REPO_NAME="headway-pipelines"
IMAGE_NAME="headway-training"

# Try to get git commit hash for tag, otherwise use timestamp
if git rev-parse --short HEAD >/dev/null 2>&1; then
    TAG="git-$(git rev-parse --short HEAD)"
else
    TAG="date-$(date +%Y%m%d-%H%M%S)"
fi

# Artifact Registry URI Base
REPO_URI="${ARTIFACT_REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"
IMAGE_URI="${REPO_URI}:${TAG}"

# Check for skip build and missing image
if [ "$SKIP_BUILD" = true ]; then
    if ! gcloud artifacts docker images describe ${IMAGE_URI} &>/dev/null; then
        echo "Image ${IMAGE_URI} does not exist."
        echo "Searching for latest available image..."
        
        # Find latest tag using tags list sorted by version creation time
        LATEST_TAG=$(gcloud artifacts docker tags list ${REPO_URI} \
            --sort-by=~version.createTime \
            --limit=1 \
            --format="value(tag)")
        
        if [ -n "$LATEST_TAG" ]; then
            TAG=$LATEST_TAG
            IMAGE_URI="${REPO_URI}:${TAG}"
            echo "Falling back to latest image: ${IMAGE_URI}"
        else
            echo "Error: No existing images found in ${REPO_URI}. Cannot skip build."
            exit 1
        fi
    fi
fi

PIPELINE_ROOT="gs://${PROJECT_ID}-pipelines/headway-prediction"

echo "========================================================================"
echo "DEPLOYING TO VERTEX AI"
echo "Project: $PROJECT_ID"
echo "Region:  $REGION"
echo "Image:   $IMAGE_URI"
echo "========================================================================"

# 1. Create Artifact Registry Repository (if not exists)
echo -e "\n[Step 1/4] Checking Artifact Registry..."
if ! gcloud artifacts repositories describe ${REPO_NAME} --location=${ARTIFACT_REGION} --project=${PROJECT_ID} &>/dev/null; then
    echo "Creating repository ${REPO_NAME}..."
    gcloud artifacts repositories create ${REPO_NAME} \
        --repository-format=docker \
        --location=${ARTIFACT_REGION} \
        --description="Headway Prediction Pipeline Images"
else
    echo "Repository ${REPO_NAME} exists."
fi

# 2. Check for existing image
echo -e "\n[Step 2/4] Checking for existing image..."
BUILD_IMAGE=true

if [ "$SKIP_BUILD" = true ]; then
    echo "Skipping build as requested via --skip-build flag."
    BUILD_IMAGE=false
elif gcloud artifacts docker images describe ${IMAGE_URI} &>/dev/null; then
    echo "Image ${IMAGE_URI} already exists."
    read -p "Do you want to rebuild the image anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping build step. Using existing image."
        BUILD_IMAGE=false
    fi
fi

# 3. Build and Push Image (Cloud Build)
if [ "$BUILD_IMAGE" = true ]; then
    echo -e "\n[Step 3/4] Building and Pushing Image with Cloud Build..."
    # Build from ml_pipelines directory
    cd "${PROJECT_ROOT}/ml_pipelines"
    
    # Use Cloud Build
    gcloud builds submit --tag ${IMAGE_URI} .    
else
    echo -e "\n[Step 3/4] Skipped Build."
fi

# 4. Compile and Run Pipeline
echo -e "\n[Step 4/4] Submitting Pipeline..."
# We use a temporary python script to compile and submit the pipeline
# passing the dynamically generated IMAGE_URI
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
export TENSORFLOW_IMAGE_URI="${IMAGE_URI}"
python3 -c "
import os
from kfp import compiler
from google.cloud import aiplatform
from ml_pipelines.training_pipeline import training_pipeline

# Compile
compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path='headway_pipeline.json'
)

# Submit
aiplatform.init(
    project='${PROJECT_ID}',
    location='${REGION}',
    staging_bucket='${PIPELINE_ROOT}'
)

job = aiplatform.PipelineJob(
    display_name='headway-training-${TAG}',
    template_path='headway_pipeline.json',
    pipeline_root='${PIPELINE_ROOT}',
    parameter_values={
        'project_id': '${PROJECT_ID}',
        'vertex_location': '${REGION}',
        'tensorboard_root': '${PIPELINE_ROOT}/tensorboard',
        'epochs': 50
    },
    enable_caching=True
)

job.submit()
print(f'Pipeline submitted: {job.resource_name}')
"

echo "========================================================================"
echo "DEPLOYMENT COMPLETE"
echo "Track your pipeline at: https://console.cloud.google.com/vertex-ai/pipelines/runs?project=${PROJECT_ID}"
echo "========================================================================"
