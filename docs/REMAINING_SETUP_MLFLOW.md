# Remaining Setup: Production MLflow on GCP

**Objective**: Deploy a secure, serverless MLflow Tracking Server on Google Cloud Run backed by Cloud SQL and GCS. This enables persistent experiment tracking for Vertex AI training jobs.

## Prerequisites
Open a terminal and set your environment variables before running any commands:

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export DB_INSTANCE_NAME="mlflow-backend"
export DB_NAME="mlflow_db"
export DB_USER="mlflow_user"
# Generate or Set a strong password
export DB_PASS="YOUR_SECURE_PASSWORD" 
```

---

## Step 1: Database Infrastructure (Cloud SQL)
*If you haven't run this yet, execute the following to provision the database. If you already ran it, skip to Step 2.*

```bash
# 1. Enable APIs
gcloud services enable run.googleapis.com sqladmin.googleapis.com artifactregistry.googleapis.com

# 2. Create Instance (Approx. 5-7 mins)
gcloud sql instances create $DB_INSTANCE_NAME \
    --project=$PROJECT_ID \
    --database-version=POSTGRES_14 \
    --tier=db-f1-micro \
    --region=$REGION \
    --availability-type=ZONAL \
    --storage-auto-increase

# 3. Create Database & User
gcloud sql databases create $DB_NAME --instance=$DB_INSTANCE_NAME
gcloud sql users create $DB_USER --instance=$DB_INSTANCE_NAME --password=$DB_PASS
```

---

## Step 2: Build MLflow Docker Image
*We need a custom image that contains MLflow and the Cloud SQL connectors.*

1. Create a `Dockerfile.mlflow` in the root directory:

```dockerfile
FROM python:3.11-slim

RUN pip install mlflow[extras] psycopg2-binary google-cloud-storage google-cloud-logging

# Standard MLflow Server Port
EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql://${DB_USER}:${DB_PASS}@/cloudsql/${INSTANCE_CONNECTION_NAME}/${DB_NAME}", \
     "--default-artifact-root", "${BUCKET_URI}"]
```

2. Build and Push to Artifact Registry:

```bash
# Get your connection name
export INSTANCE_CONNECTION_NAME=$(gcloud sql instances describe $DB_INSTANCE_NAME --format="value(connectionName)")
export BUCKET_URI="gs://${PROJECT_ID}-zenml-store/mlruns"

# Build
docker build -t gcr.io/${PROJECT_ID}/mlflow-server -f Dockerfile.mlflow .

# Push
docker push gcr.io/${PROJECT_ID}/mlflow-server
```

---

## Step 3: Deploy to Cloud Run
*Deploy the container as a serverless HTTPS endpoint.*

```bash
gcloud run deploy mlflow-server \
    --image gcr.io/${PROJECT_ID}/mlflow-server \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --add-cloudsql-instances $INSTANCE_CONNECTION_NAME \
    --set-env-vars DB_USER=$DB_USER \
    --set-env-vars DB_PASS=$DB_PASS \
    --set-env-vars DB_NAME=$DB_NAME \
    --set-env-vars INSTANCE_CONNECTION_NAME=$INSTANCE_CONNECTION_NAME \
    --set-env-vars BUCKET_URI=$BUCKET_URI
```

*Note: For maximum security, command modification is needed to inject the `DB_PASS` purely as run-time arguments or secrets, but for this setup, env vars are sufficient.*

**Output**: You will get a Service URL (e.g., `https://mlflow-server-xyz.a.run.app`).

---

## Step 4: Wire up ZenML
*Connect your local stack and Vertex pipeline to this new server.*

```bash
export MLFLOW_TRACKING_URI="https://<YOUR-CLOUD-RUN-URL>"

zenml experiment-tracker update mlflow_tracker \
    --tracking_uri=$MLFLOW_TRACKING_URI \
    --tracking_username=username \
    --tracking_password=password
```

---

## Step 5: Run the Training Pipeline
*Now everything is ready for the A100 training job.*

```bash
python -m mlops_pipeline.run --data_path gs://${PROJECT_ID}-zenml-store/data/training_data.parquet
```

## Cost Saving Tips (End of Day)
To pause the database billing ($0.03/hour) when not in use:

```bash
gcloud sql instances patch $DB_INSTANCE_NAME --activation-policy=NEVER
```

To resume:
```bash
gcloud sql instances patch $DB_INSTANCE_NAME --activation-policy=ALWAYS
```
