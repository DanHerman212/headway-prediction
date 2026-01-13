#!/bin/bash
# =============================================================================
# Test Backfill Script (7 days)
# =============================================================================
# Runs a 7-day test of the backfill pipeline, then cleans up.
# After validation, run the full backfill with: ./scripts/run_backfill.sh
#
# Usage:
#   ./scripts/run_backfill_test.sh
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

# Override date range for test (last 7 days)
TEST_END_DATE=$(date -v-1d +%Y-%m-%d)  # Yesterday
TEST_START_DATE=$(date -v-8d +%Y-%m-%d)  # 8 days ago (7 full days)

# Export for Python scripts
export GCS_BUCKET_NAME="${GCP_BUCKET}"
export BQ_DATASET_ID="mta_raw"
export BQ_TABLE_ID="raw"

SCRIPT_DIR="infrastructure/docker/ingestion"
SQL_DIR="pipelines/sql"

echo "=========================================="
echo "MTA Backfill TEST (7 days)"
echo "=========================================="
echo "Project:    ${GCP_PROJECT_ID}"
echo "Bucket:     ${GCP_BUCKET}"
echo "Test range: ${TEST_START_DATE} to ${TEST_END_DATE}"
echo "=========================================="
echo ""
read -p "Press Enter to start test, or Ctrl+C to cancel..."

# -----------------------------------------------------------------------------
# Step 0: Download GTFS (one-time, for stops table)
# -----------------------------------------------------------------------------
echo ""
echo "[Step 0] Downloading GTFS static data..."
cd "${SCRIPT_DIR}"
python3 download_gtfs.py
cd - > /dev/null

echo ""
echo "[Step 0b] Loading stops to BigQuery..."
bq load --source_format=CSV --skip_leading_rows=1 --autodetect --replace \
    "${GCP_PROJECT_ID}:mta_raw.stops" \
    "gs://${GCP_BUCKET}/raw/gtfs/stops.txt"

# -----------------------------------------------------------------------------
# Step 1: Download 7 days of arrivals
# -----------------------------------------------------------------------------
echo ""
echo "[Step 1] Downloading 7 days of arrival data..."

cd "${SCRIPT_DIR}"
python3 download_historical_data.py \
    --start_date "${TEST_START_DATE}" \
    --end_date "${TEST_END_DATE}"
cd - > /dev/null

# -----------------------------------------------------------------------------
# Step 2: Delete trips files
# -----------------------------------------------------------------------------
echo ""
echo "[Step 2] Cleaning up trips files from GCS..."
cd "${SCRIPT_DIR}"
FORCE_DELETE=true python3 delete_trips_files.py
cd - > /dev/null

# -----------------------------------------------------------------------------
# Step 3: Load to BigQuery
# -----------------------------------------------------------------------------
echo ""
echo "[Step 3] Loading CSVs to BigQuery..."
cd "${SCRIPT_DIR}"
python3 load_to_bigquery_monthly.py --year 2026 --month 1
cd - > /dev/null

# -----------------------------------------------------------------------------
# Step 4: Run SQL transforms
# -----------------------------------------------------------------------------
echo ""
echo "[Step 4] Running SQL transforms..."

echo "  [4a] Cleaning arrivals..."
# The cleansation SQL:
# - Reads from mta_raw.raw and mta_raw.stops
# - Writes to mta_transformed.clean
# So we need to replace params for both datasets
cat "${SQL_DIR}/02_data_cleansation.sql" | \
    sed "s/{{ params.project_id }}/${GCP_PROJECT_ID}/g" | \
    sed "s/{{ params.dataset_id }}.clean/mta_transformed.clean/g" | \
    sed "s/{{ params.dataset_id }}.raw/mta_raw.raw/g" | \
    sed "s/{{ params.dataset_id }}.stops/mta_raw.stops/g" | \
    bq query --use_legacy_sql=false --project_id="${GCP_PROJECT_ID}"

echo "  [4b] Computing headways for A/C/E lines..."
sed "s/{{ params.project_id }}/${GCP_PROJECT_ID}/g; s/{{ params.dataset_id }}/mta_transformed/g" \
    "${SQL_DIR}/03_ml_headways_all_nodes.sql" | \
    bq query --use_legacy_sql=false --project_id="${GCP_PROJECT_ID}"

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Test Complete - Validation"
echo "=========================================="
echo ""
echo "Checking row counts..."

RAW_COUNT=$(bq query --use_legacy_sql=false --format=csv --quiet \
    "SELECT COUNT(*) FROM \`${GCP_PROJECT_ID}.mta_raw.raw\`" | tail -1)
echo "  mta_raw.raw:              ${RAW_COUNT} rows"

CLEAN_COUNT=$(bq query --use_legacy_sql=false --format=csv --quiet \
    "SELECT COUNT(*) FROM \`${GCP_PROJECT_ID}.mta_transformed.clean\`" | tail -1)
echo "  mta_transformed.clean:    ${CLEAN_COUNT} rows"

HEADWAYS_COUNT=$(bq query --use_legacy_sql=false --format=csv --quiet \
    "SELECT COUNT(*) FROM \`${GCP_PROJECT_ID}.mta_transformed.headways_all_nodes\`" | tail -1)
echo "  mta_transformed.headways: ${HEADWAYS_COUNT} rows"

echo ""
echo "Sample headways data:"
bq query --use_legacy_sql=false --project_id="${GCP_PROJECT_ID}" --format=pretty \
    "SELECT node_id, stop_name, route_id, direction, 
            FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', arrival_time) as arrival,
            ROUND(headway_minutes, 1) as headway_min
     FROM \`${GCP_PROJECT_ID}.mta_transformed.headways_all_nodes\`
     ORDER BY arrival_time DESC
     LIMIT 10"

echo ""
echo "=========================================="
echo "Cleanup Options"
echo "=========================================="
echo ""
echo "If the test looks good, clean up with:"
echo ""
echo "  1. Delete GCS test data:"
echo "     gsutil -m rm -r gs://${GCP_BUCKET}/decompressed/"
echo ""
echo "  2. Truncate BigQuery tables:"
echo "     bq query --use_legacy_sql=false 'TRUNCATE TABLE \`${GCP_PROJECT_ID}.mta_raw.raw\`'"
echo "     bq query --use_legacy_sql=false 'TRUNCATE TABLE \`${GCP_PROJECT_ID}.mta_transformed.clean\`'"
echo "     bq query --use_legacy_sql=false 'TRUNCATE TABLE \`${GCP_PROJECT_ID}.mta_transformed.headways_all_nodes\`'"
echo ""
echo "  3. Run full backfill:"
echo "     ./scripts/run_backfill.sh"
echo ""
read -p "Run cleanup now? (y/N): " CLEANUP

if [[ "$CLEANUP" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Cleaning up..."
    
    echo "  Deleting GCS data..."
    gsutil -m rm -r "gs://${GCP_BUCKET}/decompressed/" 2>/dev/null || true
    
    echo "  Truncating BigQuery tables..."
    bq query --use_legacy_sql=false --quiet \
        "TRUNCATE TABLE \`${GCP_PROJECT_ID}.mta_raw.raw\`"
    bq query --use_legacy_sql=false --quiet \
        "TRUNCATE TABLE \`${GCP_PROJECT_ID}.mta_transformed.clean\`"
    bq query --use_legacy_sql=false --quiet \
        "TRUNCATE TABLE \`${GCP_PROJECT_ID}.mta_transformed.headways_all_nodes\`"
    
    echo ""
    echo "✅ Cleanup complete!"
    echo ""
    echo "Ready for full backfill. Run:"
    echo "  ./scripts/run_backfill.sh"
else
    echo ""
    echo "Cleanup skipped. Run the commands above manually when ready."
fi
