#!/bin/bash
# =============================================================================
# Build and Deploy Script
# =============================================================================
# Builds Docker image and deploys Cloud Run job for weekly pipeline.
# Also sets up Cloud Scheduler to trigger weekly on Tuesday 2pm ET.
#
# Usage:
#   ./scripts/build_and_deploy.sh
# =============================================================================

set -e

# Load environment variables from .env if present
if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-east1}"
BUCKET="${GCP_BUCKET:-}"
REPO_NAME="mta-pipeline"
IMAGE_NAME="weekly-pipeline"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# -----------------------------------------------------------------------------
# Validate Configuration
# -----------------------------------------------------------------------------
if [[ -z "$PROJECT_ID" ]]; then
    log_error "GCP_PROJECT_ID not set. Please configure .env file."
    exit 1
fi

if [[ -z "$BUCKET" ]]; then
    log_error "GCP_BUCKET not set. Please configure .env file."
    exit 1
fi

REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"

echo ""
echo "=========================================="
echo "MTA Weekly Pipeline - Build & Deploy"
echo "=========================================="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Bucket:   ${BUCKET}"
echo "Registry: ${REGISTRY}"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# Copy SQL files to Docker context
# -----------------------------------------------------------------------------
log_info "Copying SQL files to Docker context..."

mkdir -p pipelines/sql
cp pipelines/sql/*.sql pipelines/sql/

# -----------------------------------------------------------------------------
# Build Docker image
# -----------------------------------------------------------------------------
log_info "Building Docker image..."

docker build -t "${REGISTRY}/${IMAGE_NAME}:latest" \
    pipelines/

# -----------------------------------------------------------------------------
# Push to Artifact Registry
# -----------------------------------------------------------------------------
log_info "Configuring Docker for Artifact Registry..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

log_info "Pushing image to Artifact Registry..."
docker push "${REGISTRY}/${IMAGE_NAME}:latest"

# -----------------------------------------------------------------------------
# Deploy Cloud Run Job
# -----------------------------------------------------------------------------
log_info "Deploying Cloud Run Job: weekly-pipeline..."

gcloud run jobs deploy weekly-pipeline \
    --image="${REGISTRY}/${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --task-timeout=1800 \
    --memory=1Gi \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${BUCKET},BQ_DATASET_RAW=mta_raw,BQ_DATASET_TRANSFORMED=mta_transformed" \
    --max-retries=1 \
    --quiet

log_info "Cloud Run Job deployed successfully!"

# -----------------------------------------------------------------------------
# Create Cloud Scheduler Job
# -----------------------------------------------------------------------------
log_info "Setting up Cloud Scheduler (Tuesday 2pm ET)..."

# Get project number for compute service account
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
SCHEDULER_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Grant Cloud Run Invoker role to the scheduler service account
log_info "Ensuring scheduler service account has Cloud Run Invoker role..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SCHEDULER_SA}" \
    --role="roles/run.invoker" \
    --condition=None \
    --quiet 2>/dev/null || true

# Check if scheduler job exists
if gcloud scheduler jobs describe weekly-pipeline-trigger \
    --location="${REGION}" \
    --project="${PROJECT_ID}" &>/dev/null; then
    
    log_warn "Scheduler job exists, updating..."
    gcloud scheduler jobs update http weekly-pipeline-trigger \
        --location="${REGION}" \
        --project="${PROJECT_ID}" \
        --schedule="0 14 * * 2" \
        --time-zone="America/New_York" \
        --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/weekly-pipeline:run" \
        --http-method=POST \
        --oauth-service-account-email="${SCHEDULER_SA}" \
        --quiet
else
    log_info "Creating scheduler job..."
    gcloud scheduler jobs create http weekly-pipeline-trigger \
        --location="${REGION}" \
        --project="${PROJECT_ID}" \
        --schedule="0 14 * * 2" \
        --time-zone="America/New_York" \
        --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/weekly-pipeline:run" \
        --http-method=POST \
        --oauth-service-account-email="${SCHEDULER_SA}" \
        --quiet
fi

log_info "Cloud Scheduler configured: Every Tuesday at 2pm ET"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Cloud Run Job:"
echo "  Name:     weekly-pipeline"
echo "  Region:   ${REGION}"
echo "  Image:    ${REGISTRY}/${IMAGE_NAME}:latest"
echo ""
echo "Cloud Scheduler:"
echo "  Name:     weekly-pipeline-trigger"
echo "  Schedule: Every Tuesday at 2:00 PM ET"
echo "  Cron:     0 14 * * 2"
echo ""
echo "Manual execution:"
echo "  gcloud run jobs execute weekly-pipeline --region=${REGION}"
echo ""
echo "View logs:"
echo "  gcloud run jobs executions list --job=weekly-pipeline --region=${REGION}"
echo ""
