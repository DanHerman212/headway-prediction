#!/bin/bash
# =============================================================================
# start_zenml.sh — ZenML Server container entrypoint
#
# Constructs ZENML_STORE_URL at RUNTIME from environment variables
# injected by Cloud Run (including DB_PASSWORD from Secret Manager).
#
# Required env vars (set via Cloud Run --set-env-vars / --set-secrets):
#   DB_USER, DB_PASSWORD, ZENML_DB, INSTANCE_CONNECTION_NAME
# =============================================================================
set -e

# Validate all required env vars are present
for VAR in DB_PASSWORD DB_USER ZENML_DB INSTANCE_CONNECTION_NAME; do
  if [ -z "${!VAR}" ]; then
    echo "FATAL: ${VAR} environment variable is not set."
    exit 1
  fi
done

# Build the connection string using TCP via the Cloud SQL Auth Proxy.
# The proxy sidecar is configured with =tcp:3306 in deploy_zenml.sh,
# so it listens on 127.0.0.1:3306 inside the container.
#
# WHY TCP INSTEAD OF UNIX SOCKET:
# ZenML v0.93+ has a strict URL validator that only allows SSL-related
# query parameters (ssl, ssl_ca, ssl_cert, ssl_key, ssl_verify_server_cert).
# It rejects ?unix_socket=... with a validation error. TCP avoids this
# because no query parameters are needed: mysql://user:pass@host:port/db
#
# NOTE: ZenML only accepts 'mysql://' scheme (not mysql+pymysql://).
# It internally rewrites to mysql+pymysql://, so pymysql must be installed.
export ZENML_STORE_URL="mysql://${DB_USER}:${DB_PASSWORD}@127.0.0.1:3306/${ZENML_DB}"
export ZENML_STORE_TYPE="sql"

echo "========================================"
echo " Starting ZenML Server"
echo "========================================"
echo " Database:  ${ZENML_DB}"
echo " Cloud SQL: ${INSTANCE_CONNECTION_NAME}"
echo " Store:     sql (MySQL via pymysql + TCP proxy on 127.0.0.1:3306)"
echo "========================================"

# NOTE: First startup performs schema migration (~30-60s).
# Cloud Run may retry the health check — this is expected behavior.
exec uvicorn zenml.zen_server.zen_server_api:app \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 1
