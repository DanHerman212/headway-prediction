# GCP Data Engineering & Orchestration Plan

**Date:** February 3, 2026  
**Objective:** Establish a robust, serverless pipeline to capture larger datasets and prepare for scalable feature engineering.

## 1. Architecture Overview

We are moving away from ad-hoc notebook execution and heavy orchestrators (Cloud Composer) to a lightweight, serverless architectures using **Google Cloud Workflows** and **Cloud Run Jobs**.

### Component Selection
*   **Orchestrator:** **Cloud Workflows**. Low-latency, event-driven, effectively zero maintenance.
*   **Compute:** **Cloud Run Jobs**. Perfect for containerized Python scripts (batch ingestion) that run to completion. Better than Cloud Functions for this usage due to longer timeouts (up to 24h) and no need for HTTP wrappers.
*   **Storage (Stage):** Google Cloud Storage (GCS) buckets for raw inputs (GTFS/Historical JSONs).
*   **Warehouse:** BigQuery for raw data storage.
*   **Transformation (Future):** **Cloud Dataflow (Apache Beam)** for scalable ETL and feature engineering (unified batch/stream).

---

## 2. Phase 1: Ingestion Workflow Implementation
*Focus: Automating the existing 4 Python scripts in `batch_ingestion/python`.*

### The Scripts
1.  `download_gtfs.py`: Fetches static schedule data.
2.  `download_historical_data.py`: Fetches realtime feed archives.
3.  `load_to_bigquery_monthly.py`: Loads processed/raw data into the Warehouse.
4.  `delete_trips_files.py`: Cleanup of local/interim files.

### Implementation Steps

#### A. Containerization
Instead of managing python environments on VMs, we will package the scripts into a single Docker image.
*   **Action:** Create a `Dockerfile` in `batch_ingestion/`.
*   **Strategy:** Single image, using input arguments to select which script to run (e.g., `ENTRYPOINT ["python"]` and passing script name as arg).

#### B. Cloud Run Jobs Setup
We will define 4 distinct Jobs (or 1 parameterized Job) using the Docker image.
1.  `ingest-gtfs-job`
2.  `ingest-historical-job`
3.  `load-bq-job`
4.  `cleanup-job`

#### C. Cloud Workflows Logic
We will write a `workflow.yaml` to orchestrate the dependencies.
*   **Parallel Branch:** Run `download_gtfs` and `download_historical_data` simultaneously to save time.
*   **Barrier:** Wait for both downloads to complete.
*   **Sequential Step:** Run `load_to_bigquery_monthly`.
*   **Final Step:** Run `delete_trips_files` (only if BQ load succeeds).

### Proposed Workflow YAML Structure
```yaml
main:
    steps:
    - init:
        assign:
            - project: ${sys.get_env("GOOGLE_CLOUD_PROJECT_ID")}
            - location: "us-central1"
    - run_downloads_parallel:
        parallel:
            branches:
            - download_gtfs:
                call: googleapis.run.v1.namespaces.jobs.run
                args:
                    name: "namespaces/${project}/jobs/ingest-gtfs-job"
                    location: ${location}
                    waitForJobToComplete: true
            - download_historical:
                call: googleapis.run.v1.namespaces.jobs.run
                args:
                    name: "namespaces/${project}/jobs/ingest-historical-job"
                    location: ${location}
                    waitForJobToComplete: true
    - load_to_bq:
        call: googleapis.run.v1.namespaces.jobs.run
        args:
            name: "namespaces/${project}/jobs/load-bq-job"
            location: ${location}
            waitForJobToComplete: true
    - cleanup:
        call: googleapis.run.v1.namespaces.jobs.run
        args:
            name: "namespaces/${project}/jobs/cleanup-job"
            location: ${location}
            waitForJobToComplete: true
```

---

## 3. Phase 2: Transformation & Feature Engineering (Dataflow)
*Focus: Quality and Consistency.*

Once the raw data is reliably landing in BigQuery (or GCS), we will insert a Transformation layer before the final ML datasets are created.

### Transition to Apache Beam (Dataflow)
*   **Goal:** Write transformation logic **once** that applies to both:
    1.  **Batch Backfill:** Processing the massive historical dump we are about to create.
    2.  **Streaming Inference:** Processing live events from the MTA API in real-time.
*   **Current Issue:** The current Pandas-based logic is hard to scale and cannot easily process a stream.

### ETL & Feature Engineering Tasks
1.  **Data Quality Improvements:** 
    *   Handle the `time_idx` continuity in the pipeline, not in the model training script.
    *   Standardize "Express" representation (imputing missing stops + boolean flags) during this stage.
2.  **Event Windowing:**
    *   Calculate `preceding_train_gap` and `local_train_density` using Beam's Windowing functions.
    *   This ensures the model receives pre-calculated complex features, reducing the "learning burden" on the TFT.

---

## 4. Action Plan for Tomorrow
1.  **Review Scripts:** Ensure the 4 python scripts accept configuration via Environment Variables (for Cloud Run compatibility).
2.  **Dockerize:** Write the `Dockerfile` and build the image.
3.  **Deploy Infrastructure:** Apply Terraform (or gcloud commands) to create the Bucket, BQ Dataset, and Cloud Run Jobs.
4.  **Deploy Workflow:** Upload and Execute the `workflow.yaml` to test a full cycle.
