#!/bin/bash
# =============================================================================
# Historical Backfill Script
# =============================================================================
# One-time script to backfill 6 months of historical data locally.
# Run this from your terminal (not containerized).
#
# Prerequisites:
#   - Python 3.x with google-cloud-storage, google-cloud-bigquery
#   - gcloud CLI configured with your project
#   - .env file configured
#
# Usage:
#   ./scripts/run_backfill.sh
# =============================================================================

set -e

# Load environment variables
if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "ERROR: .env file not found. Copy .env.example to .env and configure."
    exit 1
fi

# Validate required variables
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID not set}"
: "${GCP_BUCKET:?GCP_BUCKET not set}"
: "${ARRIVALS_START_DATE:?ARRIVALS_START_DATE not set}"
: "${ARRIVALS_END_DATE:?ARRIVALS_END_DATE not set}"

# Export for Python scripts
export GCS_BUCKET_NAME="${GCP_BUCKET}"
export BQ_DATASET_ID="mta_raw"
export BQ_TABLE_ID="raw"

SCRIPT_DIR="infrastructure/docker/ingestion"
SQL_DIR="pipelines/sql"

echo "=========================================="
echo "MTA Historical Backfill"
echo "=========================================="
echo "Project:    ${GCP_PROJECT_ID}"
echo "Bucket:     ${GCP_BUCKET}"
echo "Date range: ${ARRIVALS_START_DATE} to ${ARRIVALS_END_DATE}"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# Step 0: Download GTFS (one-time, for stops table)
# -----------------------------------------------------------------------------
echo "[Step 0] Downloading GTFS static data..."
cd "${SCRIPT_DIR}"
python3 download_gtfs.py
cd - > /dev/null

echo ""
echo "[Step 0] Loading stops to BigQuery..."
bq load --source_format=CSV --skip_leading_rows=1 --autodetect \
    "${GCP_PROJECT_ID}:mta_raw.stops" \
    "gs://${GCP_BUCKET}/raw/gtfs/stops.txt"

# -----------------------------------------------------------------------------
# Step 1: Download historical arrivals
# -----------------------------------------------------------------------------
echo ""
echo "[Step 1] Downloading historical arrival data..."
echo "  This will take a while (~120 files)..."
echo ""

cd "${SCRIPT_DIR}"
python3 download_historical_data.py \
    --start_date "${ARRIVALS_START_DATE}" \
    --end_date "${ARRIVALS_END_DATE}"
cd - > /dev/null

# -----------------------------------------------------------------------------
# Step 2: Delete trips files (not needed, can cause issues)
# -----------------------------------------------------------------------------
echo ""
echo "[Step 2] Cleaning up trips files from GCS..."
cd "${SCRIPT_DIR}"
FORCE_DELETE=true python3 delete_trips_files.py
cd - > /dev/null

# -----------------------------------------------------------------------------
# Step 3: Load to BigQuery (monthly batches)
# -----------------------------------------------------------------------------
echo ""
echo "[Step 3] Loading CSVs to BigQuery..."
cd "${SCRIPT_DIR}"
python3 load_to_bigquery_monthly.py
cd - > /dev/null

# -----------------------------------------------------------------------------
# Step 4: Run SQL transforms
# -----------------------------------------------------------------------------
echo ""
echo "[Step 4] Running SQL transforms..."

echo "  [4a] Creating tables..."
bq query --use_legacy_sql=false --project_id="${GCP_PROJECT_ID}" \
    < "${SQL_DIR}/01_create_tables.sql"

echo "  [4b] Cleaning arrivals..."
# The cleansation SQL:
# - Reads from mta_raw.raw and mta_raw.stops
# - Writes to mta_transformed.clean
cat "${SQL_DIR}/02_data_cleansation.sql" | \
    sed "s/{{ params.project_id }}/${GCP_PROJECT_ID}/g" | \
    sed "s/{{ params.dataset_id }}.clean/mta_transformed.clean/g" | \
    sed "s/{{ params.dataset_id }}.raw/mta_raw.raw/g" | \
    sed "s/{{ params.dataset_id }}.stops/mta_raw.stops/g" | \
    bq query --use_legacy_sql=false --project_id="${GCP_PROJECT_ID}"

echo "  [4c] Computing headways for A/C/E lines..."
sed "s/{{ params.project_id }}/${GCP_PROJECT_ID}/g; s/{{ params.dataset_id }}/mta_transformed/g" \
    "${SQL_DIR}/03_ml_headways_all_nodes.sql" | \
    bq query --use_legacy_sql=false --project_id="${GCP_PROJECT_ID}"

# -----------------------------------------------------------------------------
# Step 5: Create stored procedures for weekly incremental updates
# -----------------------------------------------------------------------------
echo ""
echo "[Step 5] Creating stored procedures for weekly updates..."
bq query --use_legacy_sql=false --project_id="${GCP_PROJECT_ID}" \
    < "${SQL_DIR}/10_create_stored_procedures.sql"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "✅ Historical Backfill Complete!"
echo "=========================================="
echo ""
echo "Tables created:"
echo "  - mta_raw.raw (raw sensor data)"
echo "  - mta_raw.stops (GTFS stops)"
echo "  - mta_transformed.clean (parsed arrivals)"
echo "  - mta_transformed.headways (headway features)"
echo ""
echo "Stored procedures created:"
echo "  - mta_transformed.sp_clean_arrivals_incremental"
echo "  - mta_transformed.sp_compute_headways_incremental"
echo ""
echo "Next steps:"
echo "  1. Deploy weekly pipeline: ./scripts/build_and_deploy.sh"
echo "  2. Monitor scheduled runs in Cloud Console"
echo ""
