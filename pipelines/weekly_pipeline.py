#!/usr/bin/env python3
"""
Weekly Pipeline
===============
Orchestrates the weekly data refresh:
1. Download previous 7 days from subwaydata.nyc
2. Load CSVs to BigQuery (headway_dataset.raw)
3. Run SQL transforms (incremental headway computation)

This script is designed to run in a Cloud Run Job, triggered by Cloud Scheduler
every Tuesday at 2pm ET to fetch Mon-Sun of the prior week.

Usage:
    python weekly_pipeline.py

Environment Variables:
    GCP_PROJECT_ID: Google Cloud project ID
    GCP_BUCKET: Cloud Storage bucket name
    BQ_DATASET_RAW: Raw dataset (default: headway_dataset)
    BQ_DATASET_TRANSFORMED: Transformed dataset (default: headway_dataset)
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta
from google.cloud import bigquery, storage

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
BUCKET_NAME = os.environ.get("GCP_BUCKET")
DATASET_RAW = os.environ.get("BQ_DATASET_RAW", "headway_dataset")
DATASET_TRANSFORMED = os.environ.get("BQ_DATASET_TRANSFORMED", "headway_dataset")
GCS_PREFIX = "decompressed/"

# SQL files to run (in order)
SQL_DIR = "/app/sql"


def get_week_ending_date():
    """
    Calculate the week-ending date (yesterday, which is the last full day).
    If today is Tuesday Jan 20, returns Jan 19 (Sunday).
    Downloads Jan 13-19 (Mon-Sun).
    """
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    return yesterday.strftime('%Y-%m-%d')


def step1_download(week_ending: str) -> bool:
    """
    Download 7 days of arrival data from subwaydata.nyc.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Download Weekly Data")
    print("=" * 60)
    
    cmd = ["python3", "download_weekly_data.py", "--week-ending", week_ending]
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: Download failed with exit code {result.returncode}")
        return False
    
    print("✓ Download complete")
    return True


def step1b_delete_trips_files() -> bool:
    """
    Delete *_trips.csv files from GCS (not needed, can cause schema issues).
    """
    print("\n" + "=" * 60)
    print("STEP 1b: Delete Trips Files")
    print("=" * 60)
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    # List all *_trips.csv files
    blobs = list(bucket.list_blobs(prefix=GCS_PREFIX))
    trips_blobs = [b for b in blobs if b.name.endswith('_trips.csv')]
    
    if not trips_blobs:
        print("  No *_trips.csv files found")
        return True
    
    print(f"  Deleting {len(trips_blobs)} *_trips.csv files...")
    for blob in trips_blobs:
        blob.delete()
    
    print("✓ Trips files deleted")
    return True


def step2_load_to_bigquery(week_ending: str) -> bool:
    """
    Load the week's CSV files to BigQuery.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Load to BigQuery")
    print("=" * 60)
    
    # Calculate the 7 dates
    end_date = datetime.strptime(week_ending, '%Y-%m-%d')
    dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
    
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_RAW}.raw"
    
    success_count = 0
    for date_str in dates:
        year_month = date_str[:7]
        gcs_uri = f"gs://{BUCKET_NAME}/{GCS_PREFIX}{year_month}/*{date_str}*stop_times.csv"
        
        print(f"  Loading {date_str}...")
        
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=True,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            allow_jagged_rows=True,
            max_bad_records=100,
            ignore_unknown_values=True
        )
        
        try:
            load_job = client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
            load_job.result()  # Wait for completion
            print(f"    ✓ Loaded {load_job.output_rows} rows")
            success_count += 1
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print(f"\n✓ Loaded {success_count}/7 days to BigQuery")
    return success_count > 0


def step3_run_transforms(start_date: str, end_date: str) -> bool:
    """
    Run SQL transforms for the weekly date range.
    Substitutes project_id and date range parameters into SQL files.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Run SQL Transforms")
    print("=" * 60)
    print(f"  Processing: {start_date} to {end_date}")
    
    client = bigquery.Client(project=PROJECT_ID)
    
    # SQL files to run with parameter substitution
    sql_files = [
        ("Clean arrivals (incremental)", f"{SQL_DIR}/04_data_cleansation_incremental.sql"),
        ("Compute headways (incremental)", f"{SQL_DIR}/05_ml_headways_incremental.sql"),
    ]
    
    for name, sql_file in sql_files:
        print(f"  Running: {name}...")
        try:
            with open(sql_file, 'r') as f:
                sql = f.read()
            
            # Substitute parameters
            sql = sql.replace('{{ params.project_id }}', PROJECT_ID)
            sql = sql.replace('{{ params.start_date }}', start_date)
            sql = sql.replace('{{ params.end_date }}', end_date)
            
            query_job = client.query(sql)
            query_job.result()  # Wait for completion
            print(f"    ✓ Complete")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return False
    
    print("\n✓ All transforms complete")
    return True


def main():
    print("=" * 60)
    print("MTA Weekly Data Pipeline")
    print("=" * 60)
    print(f"Project:    {PROJECT_ID}")
    print(f"Bucket:     {BUCKET_NAME}")
    print(f"Started:    {datetime.now().isoformat()}")
    
    if not PROJECT_ID or not BUCKET_NAME:
        print("ERROR: GCP_PROJECT_ID and GCP_BUCKET environment variables required")
        sys.exit(1)
    
    # Calculate week ending date and date range
    week_ending = get_week_ending_date()
    end_date = datetime.strptime(week_ending, '%Y-%m-%d')
    start_date = (end_date - timedelta(days=6)).strftime('%Y-%m-%d')
    
    print(f"Week: {start_date} to {week_ending}")
    
    # Execute pipeline steps
    if not step1_download(week_ending):
        print("\n❌ Pipeline failed at Step 1: Download")
        sys.exit(1)
    
    if not step1b_delete_trips_files():
        print("\n❌ Pipeline failed at Step 1b: Delete Trips Files")
        sys.exit(1)
    
    if not step2_load_to_bigquery(week_ending):
        print("\n❌ Pipeline failed at Step 2: Load to BigQuery")
        sys.exit(1)
    
    if not step3_run_transforms(start_date, week_ending):
        print("\n❌ Pipeline failed at Step 3: SQL Transforms")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Weekly Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
