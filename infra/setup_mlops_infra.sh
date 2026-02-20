#!/bin/bash
# =============================================================================
# setup_mlops_infra.sh
#
# Provisions ALL GCP infrastructure for the Vertex AI Pipelines + MLflow deployment:
#   - APIs, Service Account, IAM bindings
#   - Cloud SQL MySQL 8.0 instance with two databases
#   - Secret Manager secret for DB password
#   - Artifact Registry for Docker images
#   - GCS bucket for ML artifacts
#
# Idempotent: safe to run multiple times.
# Usage: ./infra/setup_mlops_infra.sh
# =============================================================================
set -euo pipefail

# =============================================================================
# Configuration — edit these to match your project
# =============================================================================
PROJECT_ID="realtime-headway-prediction"
REGION="us-east1"

# Resource names
SQL_INSTANCE="mlops-metadata"
ARTIFACT_BUCKET="mlops-artifacts-${PROJECT_ID}"
SERVICE_ACCOUNT="mlops-sa"
SECRET_NAME="mlops-db-pass"
AR_REPO="mlops-images"

# Database config
ZENML_DB="zenml_db"
MLFLOW_DB="mlflow_db"
DB_USER="mlops-user"

# Derived
SA_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=========================================="
echo " MLOps Infrastructure Setup"
echo "=========================================="
echo " Project:  ${PROJECT_ID}"
echo " Region:   ${REGION}"
echo " SQL:      ${SQL_INSTANCE}"
echo " Bucket:   gs://${ARTIFACT_BUCKET}"
echo "=========================================="
echo ""

gcloud config set project "${PROJECT_ID}"

# -------------------------------------------------
# Step 1: Enable required APIs
# -------------------------------------------------
echo "--- Step 1: Enabling APIs ---"
APIS=(
  "run.googleapis.com"
  "sqladmin.googleapis.com"          # Required for Cloud SQL Auth Proxy
  "compute.googleapis.com"
  "artifactregistry.googleapis.com"
  "secretmanager.googleapis.com"
  "iam.googleapis.com"
  "cloudbuild.googleapis.com"
)
for API in "${APIS[@]}"; do
  echo "  Enabling ${API}..."
  gcloud services enable "${API}" --quiet
done
echo "✓ All APIs enabled"
echo ""

# -------------------------------------------------
# Step 2: Service Account
# -------------------------------------------------
echo "--- Step 2: Service Account ---"
if gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
  echo "✓ Service account already exists"
else
  gcloud iam service-accounts create "${SERVICE_ACCOUNT}" \
    --display-name="MLOps Server Identity"
  echo "✓ Service account created: ${SA_EMAIL}"
fi
echo ""

# -------------------------------------------------
# Step 3: IAM Bindings (project-level)
# -------------------------------------------------
echo "--- Step 3: IAM Bindings ---"
ROLES=(
  "roles/cloudsql.client"               # Connect to Cloud SQL via Auth Proxy
  "roles/secretmanager.secretAccessor"   # Read DB password at runtime
  "roles/run.invoker"                    # Allow service-to-service calls
)
for ROLE in "${ROLES[@]}"; do
  echo "  Binding ${ROLE}..."
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SA_EMAIL}" \
    --role "${ROLE}" \
    --condition=None \
    --quiet > /dev/null
done
echo "✓ IAM roles bound"
echo ""

# -------------------------------------------------
# Step 4: Cloud SQL MySQL 8.0 Instance
# -------------------------------------------------
echo "--- Step 4: Cloud SQL Instance ---"
if gcloud sql instances describe "${SQL_INSTANCE}" &>/dev/null; then
  echo "✓ Cloud SQL instance '${SQL_INSTANCE}' already exists"
else
  echo "  Creating Cloud SQL MySQL 8.0 instance..."
  echo "  (This takes ~5-10 minutes. Go grab coffee.)"
  gcloud sql instances create "${SQL_INSTANCE}" \
    --database-version=MYSQL_8_0 \
    --tier=db-f1-micro \
    --region="${REGION}" \
    --root-password="$(openssl rand -base64 16)" \
    --storage-type=HDD \
    --storage-size=10GB \
    --quiet
  echo "✓ Cloud SQL instance created"
fi
echo ""

# -------------------------------------------------
# Step 5: Create Databases
# -------------------------------------------------
echo "--- Step 5: Creating Databases ---"
for DB in "${ZENML_DB}" "${MLFLOW_DB}"; do
  if gcloud sql databases describe "${DB}" --instance="${SQL_INSTANCE}" &>/dev/null; then
    echo "  ✓ Database '${DB}' already exists"
  else
    gcloud sql databases create "${DB}" --instance="${SQL_INSTANCE}" --quiet
    echo "  ✓ Database '${DB}' created"
  fi
done
echo ""

# -------------------------------------------------
# Step 6: DB User + Secret Manager
# -------------------------------------------------
echo "--- Step 6: Database User & Password ---"
if gcloud secrets describe "${SECRET_NAME}" &>/dev/null; then
  echo "  ✓ Secret '${SECRET_NAME}' already exists — reusing"
  DB_PASS=$(gcloud secrets versions access latest --secret="${SECRET_NAME}")
else
  # Generate a random password for the database user
  DB_PASS=$(openssl rand -base64 16)
  echo -n "${DB_PASS}" | gcloud secrets create "${SECRET_NAME}" \
    --data-file=- \
    --replication-policy="automatic" \
    --quiet
  echo "  ✓ Secret created in Secret Manager"
fi

if gcloud sql users list --instance="${SQL_INSTANCE}" --format="value(name)" | grep -q "^${DB_USER}$"; then
  echo "  ✓ Database user '${DB_USER}' already exists"
else
  gcloud sql users create "${DB_USER}" \
    --instance="${SQL_INSTANCE}" \
    --password="${DB_PASS}" \
    --quiet
  echo "  ✓ Database user '${DB_USER}' created"
fi
echo ""

# -------------------------------------------------
# Step 7: Artifact Registry
# -------------------------------------------------
echo "--- Step 7: Artifact Registry ---"
if gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" &>/dev/null; then
  echo "✓ Artifact Registry '${AR_REPO}' already exists"
else
  gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Docker images for MLOps servers" \
    --quiet
  echo "✓ Artifact Registry created"
fi
echo ""

# -------------------------------------------------
# Step 8: GCS Artifact Bucket
# -------------------------------------------------
echo "--- Step 8: GCS Artifact Bucket ---"
if gsutil ls -b "gs://${ARTIFACT_BUCKET}" &>/dev/null; then
  echo "✓ Bucket gs://${ARTIFACT_BUCKET} already exists"
else
  gsutil mb -l "${REGION}" "gs://${ARTIFACT_BUCKET}"
  echo "✓ Bucket created"
fi
# Grant storage.objectAdmin specifically on this bucket (not project-wide)
gsutil iam ch "serviceAccount:${SA_EMAIL}:objectAdmin" "gs://${ARTIFACT_BUCKET}"
echo "✓ Storage permissions granted on bucket"
echo ""

# -------------------------------------------------
# Step 9: Configure Docker auth for Artifact Registry
# -------------------------------------------------
echo "--- Step 9: Docker Auth ---"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
echo "✓ Docker configured for ${REGION}-docker.pkg.dev"
echo ""

# -------------------------------------------------
# Output summary
# -------------------------------------------------
INSTANCE_CONNECTION_NAME=$(gcloud sql instances describe "${SQL_INSTANCE}" --format="value(connectionName)")

echo "=========================================="
echo " ✅ Infrastructure Setup Complete!"
echo "=========================================="
echo ""
echo " Instance Connection Name: ${INSTANCE_CONNECTION_NAME}"
echo " Service Account:          ${SA_EMAIL}"
echo " Artifact Bucket:          gs://${ARTIFACT_BUCKET}"
echo " Artifact Registry:        ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}"
echo " Secret Name:              ${SECRET_NAME}"
echo ""
echo " Next steps:"
echo "   1. ./infra/deploy_mlflow.sh"
echo "   2. ./infra/deploy_zenml.sh"
echo "   3. ./infra/register_stack.sh"
echo ""
