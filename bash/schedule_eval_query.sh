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

# Read SQL, strip comment lines
# Write params to a temp JSON file to avoid shell escaping issues
# (backticks in BQ table refs get interpreted as command substitution otherwise)
PARAMS_FILE=$(mktemp /tmp/bq_params.XXXXXX.json)
python3 -c "
import json, re
with open('$SQL_FILE') as f:
    lines = [l for l in f if not l.strip().startswith('--')]
q = ' '.join(lines).strip()
q = re.sub(r'\s+', ' ', q)
with open('$PARAMS_FILE', 'w') as out:
    json.dump({'query': q}, out)
"

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
  --params="$(cat "$PARAMS_FILE")"

rm -f "$PARAMS_FILE"

echo ""
echo "=== Scheduled query created ==="
echo "View in console: https://console.cloud.google.com/bigquery/scheduled-queries?project=$PROJECT"
