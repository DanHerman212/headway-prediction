#!/usr/bin/env python3
"""
Script: Load to BigQuery (Daily)
Purpose: Loads a specific day's stop_times CSV data from GCS into BigQuery.

Description:
    This script is designed for the Daily Incremental Workflow.
    1.  Accepts a specific date (YYYY-MM-DD).
    2.  Locates the specific `stop_times.csv` file for that date in GCS.
        - Looks in `decompressed/YYYY-MM/`
        - Matches pattern `*{date}*stop_times.csv`
    3.  Loads ONLY that file into the `raw` table in BigQuery.
    4.  This avoids re-scanning or re-loading the entire month's data and ignores `trips.csv` files.

Usage:
    python3 load_to_bigquery_daily.py --date 2025-12-27
"""

from google.cloud import bigquery
from google.cloud import storage
import os
import argparse
from datetime import datetime

# ============================================
# Configuration
# ============================================
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
DATASET_ID = os.environ.get("BQ_DATASET_ID")
TABLE_ID = os.environ.get("BQ_TABLE_ID", "raw")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
PREFIX = "decompressed/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help='Date to process (YYYY-MM-DD)')
    return parser.parse_args()

def load_daily_data(target_date_str):
    """
    Loads the stop_times CSV file for the specific date into BigQuery.
    """
    # Parse date to get Year-Month folder
    try:
        dt = datetime.strptime(target_date_str, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Date format must be YYYY-MM-DD. Got {target_date_str}")
        return

    year_month = dt.strftime('%Y-%m')
    
    # Construct GCS URI pattern
    # We look for files containing the date and 'stop_times' in the monthly folder
    # This is robust against prefixes like 'subwaydatanyc_'
    gcs_uri_pattern = f"gs://{BUCKET_NAME}/{PREFIX}{year_month}/*{target_date_str}*stop_times.csv"
    
    print(f"Preparing to load data for {target_date_str}")
    print(f"Searching for pattern: {gcs_uri_pattern}")

    client = bigquery.Client(project=PROJECT_ID)
    table_ref = client.dataset(DATASET_ID).table(TABLE_ID)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1, # Assume header
        autodetect=True,     # Auto-detect schema (consistent with monthly script)
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND, # Append to raw table
        allow_jagged_rows=True,
        max_bad_records=1000,
        ignore_unknown_values=True
    )

    try:
        # BigQuery load_table_from_uri supports wildcards
        load_job = client.load_table_from_uri(
            gcs_uri_pattern,
            table_ref,
            job_config=job_config
        )
        
        print(f"Starting job {load_job.job_id}...")
        load_job.result()  # Waits for the job to complete.

        print(f"Job finished. Loaded data for {target_date_str} into {DATASET_ID}.{TABLE_ID}.")
        
        destination_table = client.get_table(table_ref)
        print(f"Table {TABLE_ID} now has {destination_table.num_rows} rows.")

    except Exception as e:
        print(f"Error loading data: {e}")
        # If no files match the pattern, BigQuery will raise a 404 Not Found or similar error.
        # We raise it so Airflow knows the task failed.
        raise e

if __name__ == "__main__":
    args = parse_args()
    load_daily_data(args.date)
