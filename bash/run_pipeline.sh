#!/bin/bash
#
# Script: run_pipeline.sh
# Purpose: End-to-end data ingestion pipeline for MTA headway prediction
#
# This script:
#   1. Loads configuration from .env
#   2. Creates GCS bucket and BigQuery dataset if they don't exist
#   3. Runs the data ingestion pipeline in sequence:
#      - download_historical_data.py (Extract)
#      - delete_trips_files.py (Cleanup)
#      - load_to_bigquery_monthly.py (Load)
#
# Usage:
#   ./bash/run_pipeline.sh
#
# Requirements:
#   - gcloud CLI authenticated
#   - Python 3 with google-cloud-storage and google-cloud-bigquery
#

set -e  # Exit immediately on any error

# ============================================
# Configuration
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

echo "============================================"
echo "MTA Headway Prediction - Data Pipeline"
echo "============================================"
echo ""

# Load .env file
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading configuration from .env..."
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

# Validate required variables
REQUIRED_VARS=("GCP_PROJECT_ID" "GCP_REGION" "GCP_BUCKET" "BQ_DATASET" "BQ_TABLE" "ARRIVALS_START_DATE" "ARRIVALS_END_DATE")
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: Required variable $var is not set in .env"
        exit 1
    fi
done

echo ""
echo "Configuration:"
echo "  Project:    $GCP_PROJECT_ID"
echo "  Region:     $GCP_REGION"
echo "  Bucket:     $GCP_BUCKET"
echo "  Dataset:    $BQ_DATASET"
echo "  Table:      $BQ_TABLE"
echo "  Date Range: $ARRIVALS_START_DATE to $ARRIVALS_END_DATE"
echo ""

# ============================================
# Step 1: Create GCS Bucket (if not exists)
# ============================================
echo "============================================"
echo "Step 1: Checking GCS Bucket"
echo "============================================"

if gsutil ls -b "gs://$GCP_BUCKET" &>/dev/null; then
    echo "✓ Bucket gs://$GCP_BUCKET already exists"
else
    echo "Creating bucket gs://$GCP_BUCKET in $GCP_REGION..."
    gsutil mb -p "$GCP_PROJECT_ID" -l "$GCP_REGION" "gs://$GCP_BUCKET"
    echo "✓ Bucket created successfully"
fi
echo ""

# ============================================
# Step 2: Create BigQuery Dataset (if not exists)
# ============================================
echo "============================================"
echo "Step 2: Checking BigQuery Dataset"
echo "============================================"

if bq show --project_id="$GCP_PROJECT_ID" "$BQ_DATASET" &>/dev/null; then
    echo "✓ Dataset $BQ_DATASET already exists"
else
    echo "Creating dataset $BQ_DATASET in $GCP_REGION..."
    bq mk --project_id="$GCP_PROJECT_ID" --location="$GCP_REGION" --dataset "$BQ_DATASET"
    echo "✓ Dataset created successfully"
fi
echo ""

# ============================================
# Step 3: Download Historical Data
# ============================================
echo "============================================"
echo "Step 3: Downloading Historical Data"
echo "============================================"

python3 "$PROJECT_ROOT/batch_ingestion/python/download_historical_data.py" \
    --start_date "$ARRIVALS_START_DATE" \
    --end_date "$ARRIVALS_END_DATE"

echo "✓ Download complete"
echo ""

# ============================================
# Step 4: Delete Trips Files (Cleanup)
# ============================================
echo "============================================"
echo "Step 4: Cleaning Up Trips Files"
echo "============================================"

export FORCE_DELETE=true
python3 "$PROJECT_ROOT/batch_ingestion/python/delete_trips_files.py"

echo "✓ Cleanup complete"
echo ""

# ============================================
# Step 5: Load to BigQuery
# ============================================
echo "============================================"
echo "Step 5: Loading Data to BigQuery"
echo "============================================"

python3 "$PROJECT_ROOT/batch_ingestion/python/load_to_bigquery_monthly.py" \
    --start_date "$ARRIVALS_START_DATE" \
    --end_date "$ARRIVALS_END_DATE"

echo "✓ Load complete"
echo ""

# ============================================
# Summary
# ============================================
echo "============================================"
echo "Pipeline Complete!"
echo "============================================"
echo ""
echo "Data loaded to: $GCP_PROJECT_ID.$BQ_DATASET.$BQ_TABLE"
echo ""
