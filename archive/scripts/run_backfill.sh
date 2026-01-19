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
export BQ_DATASET_ID="headway_dataset"
export BQ_TABLE_ID="raw"

SCRIPT_DIR="pipelines"
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
# Step 0: Download static data (one-time, skip if already in GCS)
# -----------------------------------------------------------------------------

# Check if GTFS already downloaded
if gsutil -q stat "gs://${GCP_BUCKET}/raw/gtfs/stops.txt" 2>/dev/null; then
    echo "[Step 0a] GTFS already in GCS, skipping download..."
else
    echo "[Step 0a] Downloading GTFS static data..."
    cd "${SCRIPT_DIR}"
    python3 download_gtfs.py
    cd - > /dev/null
fi

# Check if schedules already downloaded
if gsutil -q stat "gs://${GCP_BUCKET}/raw/schedules/historic_schedules.csv" 2>/dev/null; then
    echo "[Step 0b] Schedules already in GCS, skipping download..."
else
    echo "[Step 0b] Downloading NY Open Data schedules..."
    cd "${SCRIPT_DIR}"
    python3 download_schedules.py
    cd - > /dev/null
fi

# Check if alerts already downloaded
if gsutil -q stat "gs://${GCP_BUCKET}/raw/alerts/service_alerts.csv" 2>/dev/null; then
    echo "[Step 0c] Alerts already in GCS, skipping download..."
else
    echo "[Step 0c] Downloading NY Open Data alerts..."
    cd "${SCRIPT_DIR}"
    python3 download_alerts.py
    cd - > /dev/null
fi

echo ""
echo "[Step 0d] Loading GTFS stops to BigQuery..."
bq load --source_format=CSV --skip_leading_rows=1 --autodetect --replace \
    "${GCP_PROJECT_ID}:headway_dataset.stops" \
    "gs://${GCP_BUCKET}/raw/gtfs/stops.txt"

echo "[Step 0e] Loading GTFS routes to BigQuery..."
bq load --source_format=CSV --skip_leading_rows=1 --autodetect --replace \
    "${GCP_PROJECT_ID}:headway_dataset.routes" \
    "gs://${GCP_BUCKET}/raw/gtfs/routes.txt"

echo "[Step 0f] Loading schedules to BigQuery..."
bq load --source_format=CSV --skip_leading_rows=1 --replace \
    --schema="Service_Date:STRING,Service_Code:STRING,Train_ID:STRING,Line:STRING,Trip_Line:STRING,Direction:STRING,Stop_Order:INTEGER,GTFS_Stop_ID:STRING,Arrival_Time:STRING,Departure_Time:STRING,Date_Difference:INTEGER,Track:STRING,Division:STRING,Revenue_Service:STRING,Timepoint:STRING,Trip_Type:STRING,Path_ID:STRING,Next_Trip_Type:STRING,Next_Trip_Time:STRING,Supplement_Schedule_Number:STRING,Schedule_File_Number:STRING,Origin_GTFS_Stop_ID:STRING,Destination_GTFS_Stop_ID:STRING" \
    "${GCP_PROJECT_ID}:headway_dataset.schedules" \
    "gs://${GCP_BUCKET}/raw/schedules/historic_schedules.csv"

echo "[Step 0g] Loading alerts to BigQuery..."
bq load --source_format=CSV --skip_leading_rows=1 --replace \
    --allow_quoted_newlines \
    --schema="Alert_ID:STRING,Event_ID:STRING,Update_Number:STRING,Date:STRING,Agency:STRING,Status_Label:STRING,Affected:STRING,Header:STRING,Description:STRING" \
    "${GCP_PROJECT_ID}:headway_dataset.alerts" \
    "gs://${GCP_BUCKET}/raw/alerts/service_alerts.csv"

# -----------------------------------------------------------------------------
# Step 1: Download historical arrivals (skip if data exists)
# -----------------------------------------------------------------------------
echo ""
# Check if arrivals data exists by looking for any stop_times.csv files
ARRIVALS_COUNT=$(gsutil ls "gs://${GCP_BUCKET}/decompressed/**/*stop_times.csv" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$ARRIVALS_COUNT" -gt 100 ]]; then
    echo "[Step 1] Historical arrivals already in GCS (${ARRIVALS_COUNT} files), skipping download..."
else
    echo "[Step 1] Downloading historical arrival data..."
    echo "  This will take a while (~120 files)..."
    echo ""
    cd "${SCRIPT_DIR}"
    python3 download_historical_data.py \
        --start_date "${ARRIVALS_START_DATE}" \
        --end_date "${ARRIVALS_END_DATE}"
    cd - > /dev/null
fi

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
python3 load_to_bigquery_monthly.py \
    --start_date "${ARRIVALS_START_DATE}" \
    --end_date "${ARRIVALS_END_DATE}"
cd - > /dev/null

# -----------------------------------------------------------------------------
# Step 4: Run SQL transforms
# -----------------------------------------------------------------------------
echo ""
echo "[Step 4] Running SQL transforms..."

echo "  [4a] Creating tables..."
sed "s/{{ params.project_id }}/${GCP_PROJECT_ID}/g" "${SQL_DIR}/01_create_tables.sql" | \
    bq query --use_legacy_sql=false --quiet --project_id="${GCP_PROJECT_ID}"

echo "  [4b] Cleaning arrivals..."
# SQL has explicit dataset references: headway_dataset.raw -> headway_dataset.clean
sed "s/{{ params.project_id }}/${GCP_PROJECT_ID}/g" "${SQL_DIR}/02_data_cleansation.sql" | \
    bq query --use_legacy_sql=false --quiet --project_id="${GCP_PROJECT_ID}"

echo "  [4c] Computing headways for A/C/E lines..."
# SQL has explicit dataset references: headway_dataset.clean -> headway_dataset.ml
sed "s/{{ params.project_id }}/${GCP_PROJECT_ID}/g" "${SQL_DIR}/03_ml_headways_all_nodes.sql" | \
    bq query --use_legacy_sql=false --quiet --project_id="${GCP_PROJECT_ID}"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "✅ Historical Backfill Complete!"
echo "=========================================="
echo ""
echo "Tables created:"
echo "  - headway_dataset.raw (raw sensor data)"
echo "  - headway_dataset.stops (GTFS stops)"
echo "  - headway_dataset.routes (GTFS routes)"
echo "  - headway_dataset.clean (parsed arrivals)"
echo "  - headway_dataset.ml (headways with temporal features)"
echo ""
