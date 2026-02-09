#!/bin/bash
# =============================================================================
# start_mlflow.sh — MLflow container entrypoint
#
# Constructs the backend-store-uri at RUNTIME from environment variables
# injected by Cloud Run (including the DB_PASSWORD from Secret Manager).
#
# Required env vars (set via Cloud Run --set-env-vars / --set-secrets):
#   DB_USER, DB_PASSWORD, MLFLOW_DB, INSTANCE_CONNECTION_NAME, ARTIFACT_BUCKET
# =============================================================================
set -e

# Validate all required env vars are present
for VAR in DB_PASSWORD DB_USER MLFLOW_DB INSTANCE_CONNECTION_NAME ARTIFACT_BUCKET; do
  if [ -z "${!VAR}" ]; then
    echo "FATAL: ${VAR} environment variable is not set."
    exit 1
  fi
done

# Build the connection string using Unix socket (NOT TCP)
# Format: mysql+pymysql://user:pass@/db?unix_socket=/cloudsql/PROJECT:REGION:INSTANCE
# Note the empty host field (@/) — this is mandatory for Cloud Run.
BACKEND_URI="mysql+pymysql://${DB_USER}:${DB_PASSWORD}@/${MLFLOW_DB}?unix_socket=/cloudsql/${INSTANCE_CONNECTION_NAME}"

echo "========================================"
echo " Starting MLflow Tracking Server"
echo "========================================"
echo " Database:  ${MLFLOW_DB}"
echo " Cloud SQL: ${INSTANCE_CONNECTION_NAME}"
echo " Artifacts: gs://${ARTIFACT_BUCKET}/mlflow"
echo "========================================"

exec mlflow server \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-store-uri "${BACKEND_URI}" \
  --default-artifact-root "gs://${ARTIFACT_BUCKET}/mlflow" \
  --serve-artifacts
