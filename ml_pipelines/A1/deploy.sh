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
#   ./deploy.sh [RUN_NAME]
#
# Example:
#   ./deploy.sh baseline_001

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

# Step 2: Build container using Cloud Build
echo "=========================================="
echo "Step 2: Building Container (Cloud Build)"
echo "=========================================="
echo "Submitting build to Cloud Build..."
echo ""

gcloud builds submit \
    --config=cloudbuild.yaml \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    .

echo ""
echo "Container build complete!"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"
echo "Image: ${IMAGE_URI}"
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
