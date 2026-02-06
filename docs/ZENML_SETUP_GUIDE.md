# ZenML & MLflow on Google Cloud - Complete Setup Guide

This guide details the end-to-end process for setting up a production-grade MLOps stack using ZenML as the control plane, Vertex AI as the orchestrator, and a hosted MLflow server for experiment tracking.

## 1. Prerequisites

Ensure you have the following CLI tools installed:
- **Google Cloud SDK** (`gcloud`): [Installation Guide](https://cloud.google.com/sdk/docs/install)
- **ZenML** (`zenml`): `pip install zenml`
- **Docker** (Required for building pipeline images)
- **Python 3.11** (Critical: 3.13 removes `distutils` needed by ZenML, and Pandas < 3.0.0 is required for PyTorch Forecasting Compatibility)

## 2. Environment Setup

Clean setup to avoid dependency conflicts:

```bash
# Create a fresh environment with Python 3.11
conda create -n zenml-vertex python=3.11 -y
conda activate zenml-vertex

# Install project dependencies
pip install -r mlops_pipeline/requirements.txt

# Install ZenML Integrations
zenml integration install gcp sklearn mlflow -y
```

## 3. Google Cloud Infrastructure Setup

### A. Initial Configuration
Authenticate locally so ZenML can provision/access resources:

```bash
gcloud auth app-default login
gcloud config set project <YOUR_PROJECT_ID>
gcloud config set compute/region <YOUR_REGION>  # e.g., us-central1
```

### B. Create Base Resources
You need a bucket for data/artifacts and a registry for Docker images:

```bash
# Create Artifact Bucket
gcloud storage buckets create gs://<YOUR_BUCKET_NAME> --location=<YOUR_REGION>

# Create Artifact Registry (better than Container Registry)
gcloud artifacts repositories create zenml-repo \
    --repository-format=docker \
    --location=<YOUR_REGION> \
    --description="Docker repository for ZenML Pipelines"

# Authenticate Docker to the registry
gcloud auth configure-docker <YOUR_REGION>-docker.pkg.dev
```

### C. Service Account permissions
The Service Account used by Vertex AI needs specific roles to pull images and write artifacts:
1. **Vertex AI User**
2. **Storage Object Admin**
3. **Artifact Registry Writer**
4. **Service Account User**

## 4. Deploying the MLflow Tracking Server (Production Mode)

We do not use a local `mlruns` folder for cloud pipelines. We deploy a shared MLflow server on Cloud Run backed by Cloud SQL.

### Step 1: Create Cloud SQL Instance (PostgreSQL)
```bash
gcloud sql instances create mlflow-backend \
    --database-version=POSTGRES_14 \
    --cpu=2 --memory=4GB \
    --region=<YOUR_REGION> \
    --root-password=<STRONG_PASSWORD>
```
*Create a database named `mlflow` inside this instance.*

### Step 2: Deploy to Cloud Run
Deploy the official MLflow image, pointing it to your GCS bucket for artifacts and SQL for metadata.

```bash
gcloud run deploy mlflow-server \
    --image=ghcr.io/mlflow/mlflow:v2.14.1 \
    --args="server,--backend-store-uri,postgresql+psycopg2://postgres:<PASSWORD>@/cloudsql/<PROJECT_ID>:<REGION>:mlflow-backend/mlflow,--default-artifact-root,gs://<YOUR_BUCKET_NAME>/mlflow-artifacts,--host,0.0.0.0" \
    --service-account=<YOUR_SERVICE_ACCOUNT_EMAIL> \
    --region=<YOUR_REGION> \
    --allow-unauthenticated # Or configure strict access
```
**Save the URL output by this command (e.g., `https://mlflow-server-xyz.a.run.app`).**

## 5. Registering the ZenML Cloud Stack

Now we tell ZenML how to use these resources.

### A. Register Components
```bash
# 1. Orchestrator (Vertex AI) - Runs the code
zenml orchestrator register vertex_orchestrator \
    --flavor=vertex \
    --region=<YOUR_REGION> \
    --workload_service_account=<YOUR_SERVICE_ACCOUNT_EMAIL>

# 2. Artifact Store (GCS) - Stores data/models
zenml artifact-store register gcs_store \
    --flavor=gcp \
    --path=gs://<YOUR_BUCKET_NAME>

# 3. Container Registry (Artifact Registry) - Stores Docker images
zenml container-registry register gcr_registry \
    --flavor=gcp \
    --uri=<YOUR_REGION>-docker.pkg.dev/<PROJECT_ID>/zenml-repo

# 4. Experiment Tracker (MLflow) - Logs metrics
zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri=<YOUR_CLOUDRUN_MLFLOW_URL> \
    --tracking_username=<OPTIONAL> \
    --tracking_password=<OPTIONAL>
```

### B. specific Stack Creation
Combine them into a "stack" and activate it:
```bash
zenml stack register vertex-gpu-stack \
    -o vertex_orchestrator \
    -a gcs_store \
    -c gcr_registry \
    -e mlflow_tracker

zenml stack set vertex-gpu-stack
```

## 6. Running the Pipeline

To run the pipeline on Vertex AI (which will trigger a Docker build and push):

```bash
python mlops_pipeline/run.py --config mlops_pipeline/config/pipeline_config.yaml
```

To view the results:
1. **Vertex AI Console**: See the DAG execution and logs.
2. **MLflow UI**: Visit your Cloud Run URL to see parameters, metrics, and plots (like the Rush Hour analysis).

## Troubleshooting Common Issues

*   **"Condition not met" / 403 Errors**: usually missing `Storage Object Admin` or `Artifact Registry Writer` roles on the Service Account.
*   **Python Version Mismatch**: Ensure your local python matches the base image python if using `pip` dependencies, or rely strictly on the Docker build.
*   **Pandas/Numpy Errors**: Pin `pandas<3.0.0` in your `requirements.txt` if using PyTorch Forestry or older libraries.
