#!/bin/bash
# =============================================================================
# Build and Deploy Script
# =============================================================================
# Builds Docker images and deploys Cloud Run jobs for the ingestion pipeline.
#
# Usage:
#   ./scripts/build_and_deploy.sh
# =============================================================================

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
REPO_NAME="mta-pipeline"
IMAGE_NAME="ingestion"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [[ "$PROJECT_ID" == "your-project-id" ]]; then
    log_error "Please set GCP_PROJECT_ID environment variable"
    exit 1
fi

REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"

# -----------------------------------------------------------------------------
# Build Docker image
# -----------------------------------------------------------------------------
log_info "Building Docker image..."

docker build -t "${REGISTRY}/${IMAGE_NAME}:latest" \
    infrastructure/docker/ingestion/

# -----------------------------------------------------------------------------
# Push to Artifact Registry
# -----------------------------------------------------------------------------
log_info "Pushing image to Artifact Registry..."

# Configure Docker for Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker push "${REGISTRY}/${IMAGE_NAME}:latest"

# -----------------------------------------------------------------------------
# Deploy Cloud Run Jobs
# -----------------------------------------------------------------------------
log_info "Deploying Cloud Run Jobs..."

BUCKET="${PROJECT_ID}-mta-data"

# Download Arrivals Job
log_info "  Creating download-arrivals job..."
gcloud run jobs create download-arrivals \
    --image="${REGISTRY}/${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --task-timeout=3600 \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${BUCKET}" \
    --command="python,download_arrivals.py" \
    --args="--start-date,2025-06-01,--end-date,2025-09-30" \
    --quiet 2>/dev/null || \
gcloud run jobs update download-arrivals \
    --image="${REGISTRY}/${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --task-timeout=3600 \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${BUCKET}" \
    --quiet

# Download Schedules Job (long timeout for 9GB file)
log_info "  Creating download-schedules job..."
gcloud run jobs create download-schedules \
    --image="${REGISTRY}/${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --task-timeout=7200 \
    --memory=2Gi \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${BUCKET}" \
    --command="python,download_schedules.py" \
    --quiet 2>/dev/null || \
gcloud run jobs update download-schedules \
    --image="${REGISTRY}/${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --task-timeout=7200 \
    --memory=2Gi \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${BUCKET}" \
    --quiet

# Download Alerts Job
log_info "  Creating download-alerts job..."
gcloud run jobs create download-alerts \
    --image="${REGISTRY}/${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --task-timeout=1800 \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${BUCKET}" \
    --command="python,download_alerts.py" \
    --quiet 2>/dev/null || \
gcloud run jobs update download-alerts \
    --image="${REGISTRY}/${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --task-timeout=1800 \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${BUCKET}" \
    --quiet

# Download GTFS Job
log_info "  Creating download-gtfs job..."
gcloud run jobs create download-gtfs \
    --image="${REGISTRY}/${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --task-timeout=600 \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${BUCKET}" \
    --command="python,download_gtfs.py" \
    --quiet 2>/dev/null || \
gcloud run jobs update download-gtfs \
    --image="${REGISTRY}/${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --task-timeout=600 \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${BUCKET}" \
    --quiet

# -----------------------------------------------------------------------------
# Deploy Cloud Workflow
# -----------------------------------------------------------------------------
log_info "Deploying Cloud Workflow..."

gcloud workflows deploy mta-data-pipeline \
    --source=workflows/data_pipeline.yaml \
    --location="${REGION}" \
    --quiet

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Cloud Run Jobs:"
echo "  - download-arrivals"
echo "  - download-schedules"
echo "  - download-alerts"
echo "  - download-gtfs"
echo ""
echo "Cloud Workflow:"
echo "  - mta-data-pipeline"
echo ""
echo "Run the pipeline:"
echo "  gcloud workflows run mta-data-pipeline --location=${REGION}"
echo ""
