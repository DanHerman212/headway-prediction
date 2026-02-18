#!/usr/bin/env bash
# ============================================
# Build and deploy the batch ingestion pipeline
# to Cloud Run Jobs + Cloud Workflows
# ============================================
# Usage:
#   ./infra/deploy_batch_pipeline.sh
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - .env file with all required vars
#   - APIs enabled: workflows, run, bigquery, dataflow, artifactregistry
# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment
set -a
source "${PROJECT_ROOT}/.env"
set +a

PROJECT_ID="${GCP_PROJECT_ID}"
REGION="${GCP_REGION}"
REPO="batch-ingestion"
IMAGE_INGESTION="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/batch-ingestion"
IMAGE_PIPELINE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/batch-pipeline"
TAG="$(date +%Y%m%d-%H%M%S)"

echo "============================================"
echo "Deploying Batch Ingestion Pipeline"
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Ingestion Image: ${IMAGE_INGESTION}:${TAG}"
echo "  Pipeline Image:  ${IMAGE_PIPELINE}:${TAG}"
echo "============================================"

# --- Step 1: Create Artifact Registry repo (if needed) ---
echo "--- Step 1: Ensuring Artifact Registry repo exists ---"
gcloud artifacts repositories describe "${REPO}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" 2>/dev/null || \
gcloud artifacts repositories create "${REPO}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --repository-format=docker \
  --description="Batch ingestion pipeline images"

# --- Step 2: Build and push Docker images ---
echo "--- Step 2a: Building ingestion image ---"
gcloud builds submit "${PROJECT_ROOT}" \
  --project="${PROJECT_ID}" \
  --tag="${IMAGE_INGESTION}:${TAG}" \
  --dockerfile=infra/Dockerfile.batch_ingestion \
  --timeout=600

echo "--- Step 2b: Building pipeline image ---"
gcloud builds submit "${PROJECT_ROOT}" \
  --project="${PROJECT_ID}" \
  --tag="${IMAGE_PIPELINE}:${TAG}" \
  --dockerfile=infra/Dockerfile.batch_pipeline \
  --timeout=600

# --- Step 3: Create/update Cloud Run Jobs ---
echo "--- Step 3: Creating Cloud Run Jobs ---"

# Job 1: Download historical data
gcloud run jobs create batch-download \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_INGESTION}:${TAG}" \
  --args="download_historical_data.py,--start_date,${ARRIVALS_START_DATE},--end_date,${ARRIVALS_END_DATE}" \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${GCP_BUCKET}" \
  --task-timeout=3600 \
  --max-retries=1 \
  --memory=2Gi \
  2>/dev/null || \
gcloud run jobs update batch-download \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_INGESTION}:${TAG}" \
  --args="download_historical_data.py,--start_date,${ARRIVALS_START_DATE},--end_date,${ARRIVALS_END_DATE}" \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${GCP_BUCKET}" \
  --task-timeout=3600 \
  --memory=2Gi

# Job 2: Delete trips files
gcloud run jobs create batch-delete-trips \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_INGESTION}:${TAG}" \
  --args="delete_trips_files.py" \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${GCP_BUCKET},FORCE_DELETE=true" \
  --task-timeout=600 \
  --max-retries=1 \
  2>/dev/null || \
gcloud run jobs update batch-delete-trips \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_INGESTION}:${TAG}" \
  --args="delete_trips_files.py" \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${GCP_BUCKET},FORCE_DELETE=true" \
  --task-timeout=600

# Job 3: Load to BigQuery
gcloud run jobs create batch-load-bq \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_INGESTION}:${TAG}" \
  --args="load_to_bigquery_monthly.py,--start_date,${ARRIVALS_START_DATE},--end_date,${ARRIVALS_END_DATE}" \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${GCP_BUCKET},BQ_DATASET=${BQ_DATASET},BQ_TABLE=${BQ_TABLE}" \
  --task-timeout=1800 \
  --max-retries=1 \
  --memory=2Gi \
  2>/dev/null || \
gcloud run jobs update batch-load-bq \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_INGESTION}:${TAG}" \
  --args="load_to_bigquery_monthly.py,--start_date,${ARRIVALS_START_DATE},--end_date,${ARRIVALS_END_DATE}" \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_BUCKET=${GCP_BUCKET},BQ_DATASET=${BQ_DATASET},BQ_TABLE=${BQ_TABLE}" \
  --task-timeout=1800 \
  --memory=2Gi

# Job 4: Generate training dataset (Dataflow batch pipeline)
gcloud run jobs create batch-generate-dataset \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_PIPELINE}:${TAG}" \
  --args="--project_id,${PROJECT_ID},--temp_location,gs://${GCP_BUCKET}/temp,--training_cutoff_date,${TRAINING_CUTOFF_DATE},--side_input_output,gs://realtime-headway-prediction-pipelines/side_inputs,--runner,DataflowRunner,--region,${REGION},--setup_file,/app/setup.py" \
  --set-env-vars="TRAINING_CUTOFF_DATE=${TRAINING_CUTOFF_DATE}" \
  --task-timeout=3600 \
  --max-retries=0 \
  --memory=4Gi \
  2>/dev/null || \
gcloud run jobs update batch-generate-dataset \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_PIPELINE}:${TAG}" \
  --args="--project_id,${PROJECT_ID},--temp_location,gs://${GCP_BUCKET}/temp,--training_cutoff_date,${TRAINING_CUTOFF_DATE},--side_input_output,gs://realtime-headway-prediction-pipelines/side_inputs,--runner,DataflowRunner,--region,${REGION},--setup_file,/app/setup.py" \
  --set-env-vars="TRAINING_CUTOFF_DATE=${TRAINING_CUTOFF_DATE}" \
  --task-timeout=3600 \
  --memory=4Gi

# --- Step 4: Deploy Workflow ---
echo "--- Step 4: Deploying Cloud Workflow ---"
gcloud workflows deploy batch-ingestion-pipeline \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --source="${SCRIPT_DIR}/workflow_batch_pipeline.yaml" \
  --set-env-vars="PROJECT_ID=${PROJECT_ID},REGION=${REGION},BQ_DATASET=${BQ_DATASET},TRAINING_CUTOFF_DATE=${TRAINING_CUTOFF_DATE},GCS_BUCKET=${GCP_BUCKET}"

echo ""
echo "============================================"
echo "Deployment complete!"
echo ""
echo "To execute the full pipeline:"
echo "  gcloud workflows execute batch-ingestion-pipeline \\"
echo "    --project=${PROJECT_ID} \\"
echo "    --location=${REGION}"
echo "============================================"
