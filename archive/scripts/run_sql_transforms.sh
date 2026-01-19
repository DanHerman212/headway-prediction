#!/bin/bash
# =============================================================================
# Run SQL Transforms
# =============================================================================
# Executes all SQL transform scripts in order.
#
# Usage:
#   ./scripts/run_sql_transforms.sh
# =============================================================================

set -e

PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"

echo "Running SQL transforms..."
echo ""

SQL_DIR="pipelines/sql"

for sql_file in $(ls ${SQL_DIR}/*.sql | sort); do
    echo "Executing: ${sql_file}..."
    bq query --use_legacy_sql=false --project_id="${PROJECT_ID}" < "${sql_file}"
    echo "  Done."
    echo ""
done

echo "All transforms complete!"
