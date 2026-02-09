# Running Pipelines on Google Cloud (Vertex AI)

This guide documents the process for transitioning from a local stack (running on your laptop) to a full cloud stack running on **Google Cloud Vertex AI** using ZenML and MLflow.

## 1. Prerequisites

Ensure your infrastructure is provisioned and you are connected to the ZenML server:

```bash
# Verify connection
zenml status
```

You should see your remote ZenML server URL.

## 2. One-Time Setup: Register Cloud Components

Run the following commands to configure ZenML to use Google Cloud resources for orchestration and container storage.

### A. Register the Artifact Registry
This is where ZenML will push the Docker images for your pipeline steps.

```bash
zenml container-registry register artifact_registry \
    --flavor=gcp \
    --uri=us-east1-docker.pkg.dev/realtime-headway-prediction/mlops-images
```

### B. Register the Artifact Store & Experiment Tracker
These components store your data and experiment logs.

```bash
# Register GCS Artifact Store
zenml artifact-store register gcs_store \
    --flavor=gcp \
    --path="gs://mlops-artifacts-realtime-headway-prediction/zenml"

# Register MLflow Tracker
# Note: ZenML requires credentials for remote URIs. Since our server is
# unauthenticated (public within VPC/allow-unauthenticated), we use dummy values.
zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri="https://mlflow-server-gxvzscak4q-ue.a.run.app" \
    --tracking_username=admin \
    --tracking_password=admin
```

### C. Register the Cloud Image Builder (Cloud Build)
This delegates the Docker build process to Google Cloud Build, saving local disk space and bandwidth.

```bash
zenml image-builder register gcp_image_builder --flavor=gcp
```

### D. Register the Vertex AI Orchestrator
This component is responsible for submitting your code as jobs to Vertex AI.

```bash
zenml orchestrator register vertex_orchestrator \
    --flavor=vertex \
    --location=us-east1 \
    --workload_service_account=mlops-sa@realtime-headway-prediction.iam.gserviceaccount.com
```

### C. Create the Cloud Stack
Combine these components into a new stack named `gcp_vertex_stack`.

```bash
zenml stack register gcp_vertex_stack \
    -o vertex_orchestrator \
    -a gcs_store \
    -e mlflow_tracker \
    -c artifact_registry \
    -i gcp_image_builder
```

## 3. Switching Stacks

Use this command to tell ZenML to targeting Vertex AI for future runs:

```bash
zenml stack set gcp_vertex_stack
```

To switch back to running locally (useful for debugging):
```bash
zenml stack set gcp_production_stack
```

## 4. Preparing Data

Cloud pipelines cannot read files from your local laptop. You must upload your training data to the GCS bucket.

```bash
# Upload local data to GCS
gsutil cp local_artifacts/processed_data/training_data.parquet \
    gs://mlops-artifacts-realtime-headway-prediction/data/training_data.parquet
```

## 5. Running the Pipeline

Execute the run script pointing to the GCS data path. ZenML will build the Docker image, push it to the registry, and submit the job to Vertex AI.

```bash
python mlops_pipeline/run.py \
    --data_path gs://mlops-artifacts-realtime-headway-prediction/data/training_data.parquet
```

## 6. Monitoring and Results

*   **Vertex AI Console:** Click the link provided in the terminal output to watch the pipeline execution in the Google Cloud Console.
*   **ZenML Dashboard:** View lineate, caching status, and run success.
*   **MLflow UI:** View experiments, parameters, and model metrics logged during the run.
