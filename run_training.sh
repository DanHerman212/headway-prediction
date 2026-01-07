#!/bin/bash
# =============================================================================
# Headway Prediction Training Pipeline
# =============================================================================
# This script:
# 1. Builds the training container locally for testing
# 2. Optionally runs a local test
# 3. Builds and pushes to Artifact Registry via Cloud Build
# 4. Submits the training pipeline to Vertex AI
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_ID="time-series-478616"
REGION="us-east1"
BUCKET="st-convnet-training-configuration"
REPO_NAME="headway-prediction"
IMAGE_NAME="training"
TAG=$(date +%Y%m%d-%H%M%S)

# Derived values
ARTIFACT_REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
FULL_IMAGE_URI="${ARTIFACT_REGISTRY}/${IMAGE_NAME}:${TAG}"
LOCAL_IMAGE="headway-training:local"

# GCS paths
DATA_DIR="gs://${BUCKET}/headway-prediction/data"
OUTPUT_DIR="gs://${BUCKET}/headway-prediction/outputs/${TAG}"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        echo "Error: gcloud not found. Please install Google Cloud SDK."
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Step 1: Build locally
# -----------------------------------------------------------------------------
build_local() {
    log "Building Docker image locally..."
    docker build -t ${LOCAL_IMAGE} .
    log "Local build complete: ${LOCAL_IMAGE}"
}

# -----------------------------------------------------------------------------
# Step 2: Test locally (optional)
# -----------------------------------------------------------------------------
test_local() {
    log "Running local test (Experiment 1 - Baseline)..."
    log "Note: This runs on CPU without GPU acceleration"
    
    # Create local output directory
    mkdir -p outputs/local_test
    
    # Run with local data (not GCS)
    docker run --rm \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/outputs/local_test:/app/outputs \
        -e DATA_DIR=/app/data \
        -e OUTPUT_DIR=/app/outputs \
        ${LOCAL_IMAGE} \
        --exp_id 1 \
        --data_dir /app/data \
        --output_dir /app/outputs
    
    log "Local test complete. Check outputs/local_test/"
}

# -----------------------------------------------------------------------------
# Step 3: Build and push via Cloud Build
# -----------------------------------------------------------------------------
build_cloud() {
    log "Submitting build to Cloud Build..."
    
    # Ensure Artifact Registry repo exists
    log "Ensuring Artifact Registry repository exists..."
    gcloud artifacts repositories describe ${REPO_NAME} \
        --location=${REGION} \
        --project=${PROJECT_ID} 2>/dev/null || \
    gcloud artifacts repositories create ${REPO_NAME} \
        --repository-format=docker \
        --location=${REGION} \
        --project=${PROJECT_ID} \
        --description="Headway prediction training containers"
    
    # Submit build
    gcloud builds submit \
        --config=cloudbuild.yaml \
        --project=${PROJECT_ID} \
        --substitutions=_REGION=${REGION},_REPO_NAME=${REPO_NAME},_IMAGE_NAME=${IMAGE_NAME},_TAG=${TAG}
    
    log "Cloud Build complete: ${FULL_IMAGE_URI}"
}

# -----------------------------------------------------------------------------
# Step 4: Submit pipeline to Vertex AI
# -----------------------------------------------------------------------------
submit_pipeline() {
    log "Submitting Kubeflow Pipeline to Vertex AI..."
    log "Container: ${FULL_IMAGE_URI}"
    log "Data: ${DATA_DIR}"
    log "Output: ${OUTPUT_DIR}"
    
    # Run the KFP pipeline submission script
    python3 -m src.experiments.kfp_pipeline \
        --project ${PROJECT_ID} \
        --bucket ${BUCKET} \
        --region ${REGION} \
        --container ${FULL_IMAGE_URI}
    
    log "Pipeline submitted!"
    log "Monitor at: https://console.cloud.google.com/vertex-ai/pipelines?project=${PROJECT_ID}"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  build-local     Build Docker image locally"
    echo "  test-local      Run local test (CPU only, Experiment 1)"
    echo "  build-cloud     Build and push via Cloud Build"
    echo "  submit          Submit pipeline to Vertex AI (requires cloud build first)"
    echo "  all             Full pipeline: build-local -> build-cloud -> submit"
    echo "  quick           Skip local test: build-cloud -> submit"
    echo ""
    echo "Examples:"
    echo "  $0 build-local          # Just build locally"
    echo "  $0 test-local           # Build and test locally"
    echo "  $0 all                  # Full pipeline with local test"
    echo "  $0 quick                # Skip local test, go straight to cloud"
}

check_gcloud

case "${1:-}" in
    build-local)
        build_local
        ;;
    test-local)
        build_local
        test_local
        ;;
    build-cloud)
        build_cloud
        ;;
    submit)
        submit_pipeline
        ;;
    all)
        build_local
        test_local
        build_cloud
        submit_pipeline
        ;;
    quick)
        build_local
        build_cloud
        submit_pipeline
        ;;
    *)
        usage
        exit 1
        ;;
esac

log "Done!"
