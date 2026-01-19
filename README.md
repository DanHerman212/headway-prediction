# Headway Prediction

MTA Subway headway prediction pipeline.

## Project Structure

```
headway-prediction/
├── python/          # Python ingestion and loading scripts
├── sql/             # SQL transform scripts
├── bash/            # Bash utility scripts
└── archive/         # Archived legacy code
```

## Python Scripts

- `download_historical_data.py` - Downloads and extracts MTA data to GCS
- `load_to_bigquery_monthly.py` - Loads CSV data from GCS into BigQuery

## Environment Variables

```
GCP_PROJECT_ID=your-project-id
GCS_BUCKET_NAME=your-bucket-name
BQ_DATASET_ID=mta_historical
BQ_TABLE_ID=sensor_data
```

## Quick Start

1. Set environment variables (see `.env.example`)
2. Run data ingestion: `python python/download_historical_data.py --start_date 2024-01-01 --end_date 2024-01-31`
3. Load to BigQuery: `python python/load_to_bigquery_monthly.py --year 2024 --month 1`

---
*README to be expanded with detailed documentation*
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
