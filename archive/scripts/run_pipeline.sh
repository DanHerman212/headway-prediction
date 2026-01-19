#!/bin/bash
# =============================================================================
# Run Pipeline Script
# =============================================================================
# Triggers the MTA data pipeline workflow.
#
# Usage:
#   ./scripts/run_pipeline.sh [--stage ingestion|transform|all]
# =============================================================================

set -e

PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"

# Parse arguments
STAGE="${1:-all}"

case "$STAGE" in
    ingestion)
        ARGS='{"run_ingestion": true, "run_transform": false, "run_tensor_build": false}'
        ;;
    transform)
        ARGS='{"run_ingestion": false, "run_transform": true, "run_tensor_build": false}'
        ;;
    tensors)
        ARGS='{"run_ingestion": false, "run_transform": false, "run_tensor_build": true}'
        ;;
    all)
        ARGS='{"run_ingestion": true, "run_transform": true, "run_tensor_build": true}'
        ;;
    *)
        echo "Usage: $0 [ingestion|transform|tensors|all]"
        exit 1
        ;;
esac

echo "Running MTA data pipeline (stage: $STAGE)..."
echo ""

gcloud workflows run mta-data-pipeline \
    --location="${REGION}" \
    --data="${ARGS}"
