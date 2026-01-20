#!/bin/bash
#
# Build and push Docker container for A1 training pipeline to Artifact Registry
#
# Usage:
#   ./build_container.sh

set -e

# Configuration from config.py
PROJECT_ID="realtime-headway-prediction"
REGION="us-east1"
REPOSITORY="ml-pipelines"
IMAGE_NAME="a1-training"
TAG="latest"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"

echo "=========================================="
echo "Building A1 Training Container"
echo "=========================================="
echo ""
echo "Image URI: ${IMAGE_URI}"
echo ""

# Create Artifact Registry repository if it doesn't exist
echo "Checking Artifact Registry repository..."
if ! gcloud artifacts repositories describe ${REPOSITORY} \
    --location=${REGION} \
    --project=${PROJECT_ID} &>/dev/null; then
    
    echo "Creating repository: ${REPOSITORY}"
    gcloud artifacts repositories create ${REPOSITORY} \
        --repository-format=docker \
        --location=${REGION} \
        --project=${PROJECT_ID} \
        --description="ML pipelines for headway prediction"
else
    echo "Repository ${REPOSITORY} already exists"
fi

# Configure Docker to use gcloud as credential helper
echo ""
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Build Docker image
echo ""
echo "Building Docker image..."
docker build -t ${IMAGE_URI} .

# Push to Artifact Registry
echo ""
echo "Pushing image to Artifact Registry..."
docker push ${IMAGE_URI}

echo ""
echo "=========================================="
echo "Container Build Complete!"
echo "=========================================="
echo ""
echo "Image: ${IMAGE_URI}"
echo ""
echo "Next steps:"
echo "  1. Compile pipeline: python3 pipeline.py --compile"
echo "  2. Submit pipeline: python3 pipeline.py --submit --run_name baseline_001"
echo ""
