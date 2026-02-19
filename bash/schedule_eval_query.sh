#!/usr/bin/env bash
# Schedule the hourly prediction evaluation query in BigQuery.
#
# Usage:
#   bash bash/schedule_eval_query.sh
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - BigQuery Data Transfer API enabled
#   - Tables created via: python scripts/setup_bq_monitoring.py

set -euo pipefail

PROJECT="realtime-headway-prediction"
DATASET="headway_monitoring"
LOCATION="us-east1"
DISPLAY_NAME="Headway Prediction Evaluation (Hourly)"
SCHEDULE="every 1 hours"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SQL_FILE="${SCRIPT_DIR}/batch_ingestion/sql/evaluation_query.sql"

if [[ ! -f "$SQL_FILE" ]]; then
  echo "ERROR: SQL file not found at $SQL_FILE"
  exit 1
fi

# Read SQL, collapse to single line for JSON embedding
QUERY=$(cat "$SQL_FILE" | grep -v '^--' | tr '\n' ' ' | sed 's/  */ /g')

echo "=== Scheduling evaluation query ==="
echo "  Project:  $PROJECT"
echo "  Dataset:  $DATASET"
echo "  Location: $LOCATION"
echo "  Schedule: $SCHEDULE"
echo ""

bq mk \
  --transfer_config \
  --project_id="$PROJECT" \
  --data_source=scheduled_query \
  --target_dataset="$DATASET" \
  --display_name="$DISPLAY_NAME" \
  --location="$LOCATION" \
  --schedule="$SCHEDULE" \
  --params="{\"query\":\"$QUERY\",\"write_disposition\":\"WRITE_APPEND\"}"

echo ""
echo "=== Scheduled query created ==="
echo "View in console: https://console.cloud.google.com/bigquery/scheduled-queries?project=$PROJECT"
