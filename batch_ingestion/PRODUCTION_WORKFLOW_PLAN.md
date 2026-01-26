# Batch Ingestion Production Workflow Plan

## Objective
Transition the current manual CLI-based batch ingestion process (Python scripts + SQL) into a robust, automated, enterprise-grade workflow without the overhead and cost of Cloud Composer (Airflow).

## Selected Architecture: Google Cloud Workflows + Cloud Run Jobs
**Google Cloud Workflows** was selected as the orchestration engine because:
*   **Cost-Effective:** Serverless pay-per-use model (likely <$1/month for this workload) vs. ~$450/month for Cloud Composer.
*   **Integration:** Native connectors for Cloud Run and BigQuery.
*   **Production Features:** Built-in error handling, retries, logging, and IAM security.

---

## 1. Component Design

### A. Execution Runtime (Cloud Run Jobs)
Instead of running scripts on a VM or local machine, we will package the Python scripts into a container.
*   **Container strategy:** A single Docker image containing all scripts in `batch_ingestion/python/`.
*   **Infrastructure:** Define separate **Cloud Run Jobs** for each distinct task, all using the same image but with different entrypoints/arguments.
    *   `job-download-gtfs`: Runs `download_gtfs.py`
    *   `job-download-historical`: Runs `download_historical_data.py`
    *   `job-bq-load`: Runs `load_to_bigquery_monthly.py`

### B. SQL Transformation (BigQuery Jobs)
SQL transformations (`01_create_clean_table.sql`, etc.) do not need a container. They will be executed potentially directly via the **Workflows BigQuery Connector**.

### C. Orchestrator (Workflows)
A YAML-based workflow definition will chain these steps together:
1.  **Step 1:** Call Cloud Run Job (`download_gtfs`).
2.  **Step 2:** Call Cloud Run Job (`download_historical`).
3.  **Step 3:** Call Cloud Run Job (`load_to_bigquery`).
4.  **Step 4:** Execute SQL `01_create_clean_table` (via BQ API).
5.  **Step 5:** Execute SQL `02_feature_engineering` (via BQ API).

### D. Scheduling
*   **Cloud Scheduler:** A cron job (e.g., `0 2 * * *`) that triggers the Workflow API execution.

---

## 2. Implementation Roadmap

### Phase 1: Containerization
1.  Create a `Dockerfile` in `batch_ingestion/` that installs dependencies and copies keys/scripts.
2.  Build and push image to Google Artifact Registry.

### Phase 2: Infrastructure as Code (Terraform or shell scripts)
1.  Deploy the Cloud Run Jobs definitions.
2.  Create the Workflows YAML definition file.

### Phase 3: Deployment
1.  Deploy the Workflow.
2.  Configure the Cloud Scheduler trigger.

---

## 3. Comparison with Alternatives

| Feature | Cloud Composer | Cloud Workflows | Cloud Build |
| :--- | :--- | :--- | :--- |
| **Cost** | High ($$) | Low (¢) | Low (¢) |
| **Complexity** | High (Python/DAGs) | Medium (YAML) | Low (YAML) |
| **State Management**| Excellent | Good | Minimal |
| **Best For** | Complex Data Lakes | Serverless Orchestration | CI/CD |

**Verdict:** Workflows is the optimal choice for this linear batch pipeline.
