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

# Parse arguments
SKIP_BUILD=false
RUN_NAME=""
for arg in "$@"; do
    case $arg in
        --skip-build)
            SKIP_BUILD=true
            ;;
        *)
            if [ -z "$RUN_NAME" ]; then
                RUN_NAME="$arg"
            fi
            ;;
    esac
done

# Configuration
PROJECT_ID="realtime-headway-prediction"
REGION="us-east1"
REPOSITORY="ml-pipelines"
IMAGE_NAME="a1-training"

# Get run name from argument or generate timestamp-based name
if [ -z "$RUN_NAME" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="run_${TIMESTAMP}"
fi

echo "=========================================="
echo "A1 Pipeline Deployment"
echo "=========================================="
echo ""
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Run Name: ${RUN_NAME}"
echo ""

# Define image URI
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"

# Step 1: Build container using Cloud Build
echo "=========================================="
echo "Step 1: Building Container (Cloud Build)"
echo "=========================================="

if [ "$SKIP_BUILD" = true ]; then
    echo "Skipping container build (--skip-build flag)"
    echo "Using existing image: ${IMAGE_URI}"
elif gcloud artifacts docker images describe ${IMAGE_URI} --project=${PROJECT_ID} &>/dev/null; then
    echo "Container image already exists: ${IMAGE_URI}"
    read -p "Skip rebuild? (y/n) [default: n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Using existing image: ${IMAGE_URI}"
    else
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
    fi
else
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
