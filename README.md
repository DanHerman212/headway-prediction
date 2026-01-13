# MTA Headway Prediction - Data Pipeline

End-to-end ML pipeline for predicting NYC subway headways using Graph WaveNet.

## Project Structure

```
headway-prediction/
├── docs/                          # Documentation
│   ├── data_pipeline_architecture.md
│   └── ML_Workflow_Realtime_Headway_Prediction.md
├── infrastructure/
│   └── docker/
│       └── ingestion/             # Docker containers for data ingestion
│           ├── Dockerfile
│           ├── requirements.txt
│           ├── download_arrivals.py
│           ├── download_schedules.py
│           ├── download_alerts.py
│           └── download_gtfs.py
├── pipelines/
│   ├── beam/                      # Apache Beam / Dataflow pipelines
│   │   ├── transform_arrivals.py
│   │   ├── compute_headways.py
│   │   └── build_tensors.py
│   └── sql/                       # BigQuery SQL transforms
│       ├── 01_create_raw_tables.sql
│       ├── 02_clean_arrivals.sql
│       ├── 03_join_stops.sql
│       ├── 04_compute_headways.sql
│       └── 05_compute_lateness.sql
├── workflows/                     # Cloud Workflows orchestration
│   └── data_pipeline.yaml
├── scripts/                       # Setup and utility scripts
│   ├── setup_gcp.sh
│   ├── build_and_deploy.sh
│   └── run_pipeline.sh
├── data/                          # Local data (gitignored)
├── backup/                        # Old project files (gitignored)
├── .env                           # Environment variables (gitignored)
└── .gitignore
```

## Quick Start

### 1. Set up GCP Infrastructure

```bash
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1
./scripts/setup_gcp.sh
```

### 2. Add Your Scripts

Place your existing Python download scripts in:
- `infrastructure/docker/ingestion/download_arrivals.py`

Place your SQL transforms in:
- `pipelines/sql/`

### 3. Build and Deploy

```bash
./scripts/build_and_deploy.sh
```

### 4. Run Pipeline

```bash
./scripts/run_pipeline.sh
```

## Data Sources

| Source | Size | Location |
|--------|------|----------|
| Historic Arrivals | ~10MB/day | `gs://bucket/raw/arrivals/` |
| Historic Schedules | ~9GB | `gs://bucket/raw/schedules/` |
| Historic Alerts | ~120MB | `gs://bucket/raw/alerts/` |
| Static GTFS | ~40MB | `gs://bucket/raw/gtfs/` |

## Pipeline Stages

1. **Ingestion** - Download data to Cloud Storage (Cloud Run Jobs)
2. **Transform** - Clean and compute features in BigQuery
3. **Dataset Creation** - Build ML tensors with Dataflow

## Environment Variables

Create a `.env` file or export these variables:

```bash
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1
export GCP_BUCKET=your-project-id-mta-data
```
