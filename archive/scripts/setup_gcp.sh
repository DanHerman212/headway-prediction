#!/bin/bash
# =============================================================================
# GCP Infrastructure Setup Script
# =============================================================================
# This script creates the necessary GCP resources for the MTA data pipeline.
# Run once to initialize the project infrastructure.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Billing enabled on the GCP project
#
# Usage:
#   chmod +x scripts/setup_gcp.sh
#   ./scripts/setup_gcp.sh
# =============================================================================

set -e  # Exit on error

# Load environment variables from .env if present
if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

# -----------------------------------------------------------------------------
# Configuration - EDIT THESE VALUES
# -----------------------------------------------------------------------------
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
BUCKET_NAME="${GCP_BUCKET:-${PROJECT_ID}-mta-data}"
DATASET="headway_dataset"

# -----------------------------------------------------------------------------
# Colors for output
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# -----------------------------------------------------------------------------
# Verify Configuration
# -----------------------------------------------------------------------------
echo "=========================================="
echo "GCP Infrastructure Setup"
echo "=========================================="
echo "Project ID:  $PROJECT_ID"
echo "Region:      $REGION"
echo "Bucket:      $BUCKET_NAME"
echo "=========================================="
echo ""

if [[ "$PROJECT_ID" == "your-project-id" ]]; then
    log_error "Please set GCP_PROJECT_ID environment variable or edit this script"
    exit 1
fi

read -p "Continue with setup? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# -----------------------------------------------------------------------------
# Set project
# -----------------------------------------------------------------------------
log_info "Setting active project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID"

# -----------------------------------------------------------------------------
# Enable required APIs
# -----------------------------------------------------------------------------
log_info "Enabling required GCP APIs..."

APIS=(
    "storage.googleapis.com"
    "bigquery.googleapis.com"
    "run.googleapis.com"
    "dataflow.googleapis.com"
    "workflows.googleapis.com"
    "cloudscheduler.googleapis.com"
    "cloudbuild.googleapis.com"
    "artifactregistry.googleapis.com"
)

for api in "${APIS[@]}"; do
    log_info "  Enabling $api..."
    gcloud services enable "$api" --quiet
done

# -----------------------------------------------------------------------------
# Create Cloud Storage bucket
# -----------------------------------------------------------------------------
log_info "Creating Cloud Storage bucket: $BUCKET_NAME..."

if gsutil ls -b "gs://$BUCKET_NAME" &>/dev/null; then
    log_warn "Bucket already exists, skipping..."
else
    gsutil mb -l "$REGION" "gs://$BUCKET_NAME"
fi

# Create folder structure
log_info "Creating bucket folder structure..."
echo "" | gsutil cp - "gs://$BUCKET_NAME/staging/.keep"
echo "" | gsutil cp - "gs://$BUCKET_NAME/raw/arrivals/.keep"
echo "" | gsutil cp - "gs://$BUCKET_NAME/raw/schedules/.keep"
echo "" | gsutil cp - "gs://$BUCKET_NAME/raw/alerts/.keep"
echo "" | gsutil cp - "gs://$BUCKET_NAME/raw/gtfs/.keep"
echo "" | gsutil cp - "gs://$BUCKET_NAME/ml-dataset/.keep"

# -----------------------------------------------------------------------------
# Create BigQuery dataset
# -----------------------------------------------------------------------------
log_info "Creating BigQuery dataset..."

if bq show --dataset "$PROJECT_ID:$DATASET" &>/dev/null; then
    log_warn "Dataset $DATASET already exists, skipping..."
else
    bq mk --dataset --location="$REGION" "$PROJECT_ID:$DATASET"
fi

# -----------------------------------------------------------------------------
# Create Artifact Registry repository for Docker images
# -----------------------------------------------------------------------------
log_info "Creating Artifact Registry repository..."

REPO_NAME="mta-pipeline"
if gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" &>/dev/null; then
    log_warn "Repository $REPO_NAME already exists, skipping..."
else
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="MTA data pipeline Docker images"
fi

# -----------------------------------------------------------------------------
# Create service account for pipeline
# -----------------------------------------------------------------------------
log_info "Creating service account for pipeline..."

SA_NAME="mta-pipeline-sa"
SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if gcloud iam service-accounts describe "$SA_EMAIL" &>/dev/null; then
    log_warn "Service account already exists, skipping..."
else
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="MTA Pipeline Service Account"
    
    # Wait for service account to propagate before granting roles
    log_info "Waiting for service account to propagate..."
    sleep 10
fi

# Grant necessary roles
log_info "Granting IAM roles to service account..."

ROLES=(
    "roles/storage.objectAdmin"
    "roles/bigquery.dataEditor"
    "roles/bigquery.jobUser"
    "roles/dataflow.worker"
    "roles/run.invoker"
    "roles/workflows.invoker"
)

for role in "${ROLES[@]}"; do
    log_info "  Granting $role..."
    # Retry logic for IAM binding (handles propagation delays)
    for attempt in 1 2 3; do
        if gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$SA_EMAIL" \
            --role="$role" \
            --quiet 2>/dev/null; then
            break
        else
            if [[ $attempt -lt 3 ]]; then
                log_warn "  Retrying in 5 seconds..."
                sleep 5
            else
                log_error "  Failed to grant $role after 3 attempts"
                exit 1
            fi
        fi
    done
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Resources created:"
echo "  - Cloud Storage bucket: gs://$BUCKET_NAME"
echo "  - BigQuery dataset: $DATASET"
echo "  - Artifact Registry: $REPO_NAME"
echo "  - Service Account: $SA_EMAIL"
echo ""
echo "Next steps:"
echo "  1. Add your download scripts to pipelines/"
echo "  2. Build and push Docker images"
echo "  3. Deploy Cloud Run jobs"
echo "  4. Configure Cloud Workflows"
echo ""
echo "Environment variables to set:"
echo "  export GCP_PROJECT_ID=$PROJECT_ID"
echo "  export GCP_REGION=$REGION"
echo "  export GCP_BUCKET=$BUCKET_NAME"
echo ""
