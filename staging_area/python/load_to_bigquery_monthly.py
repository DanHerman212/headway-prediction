#!/usr/bin/env python3
"""
Script: Load to BigQuery (Monthly)
Purpose: Loads CSV data from GCS into BigQuery tables, organized by month.

Description:
    This script is the "Load" part of the ETL pipeline. It:
    1.  Scans the GCS bucket for the `decompressed/` prefix.
    2.  Identifies monthly folders (e.g., `decompressed/2024-01/`).
    3.  Constructs a BigQuery Load Job for each month.
    4.  Loads the CSV files from that month's folder into the target BigQuery table (`raw`).
    5.  Uses schema auto-detection (or a defined schema) to map CSV columns to BigQuery fields.
    6.  Handles large volumes of data by processing month-by-month to avoid timeouts and manage quotas.

Usage:
    Run as a Python script or Airflow task. Requires BigQuery Job User and GCS Reader permissions.

Environment Variables:
    GCP_PROJECT_ID: Google Cloud Project ID.
    BQ_DATASET_ID: Target BigQuery Dataset.
    BQ_TABLE_ID: Target BigQuery Table.
    GCS_BUCKET_NAME: Source GCS bucket.
"""

from google.cloud import bigquery
from google.cloud import storage
import json
from datetime import datetime
import os
import argparse

# ============================================
# Configuration
# ============================================
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "time-series-478616")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "mta_historical")
TABLE_ID = os.environ.get("BQ_TABLE_ID", "sensor_data")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", f"{PROJECT_ID}-historical-data")
PREFIX = "decompressed/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, help='Year to process')
    parser.add_argument('--month', type=int, help='Month to process')
    return parser.parse_args()

args = parse_args()

# Date range to load
if args.year and args.month:
    START_YEAR = args.year
    END_YEAR = args.year
    START_MONTH = args.month
    END_MONTH = args.month
    print(f"Running in SINGLE MONTH mode: {START_YEAR}-{START_MONTH:02d}")
else:
    START_YEAR = 2021
    START_MONTH = 4
    END_YEAR = 2025
    END_MONTH = 12
    print("Running in FULL HISTORY mode")

print("="*70)

print("="*70)
print("Loading Historical Data to BigQuery - Month by Month")
print("="*70)
print(f"Project: {PROJECT_ID}")
print(f"Dataset: {DATASET_ID}")
print(f"Table: {TABLE_ID}")
print(f"Bucket: gs://{BUCKET_NAME}/{PREFIX}")
print(f"Date Range: {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}")
print("="*70 + "\n")

# Initialize clients
bq_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)

# Schema configuration
# Modified to use auto-detection since schema file is missing
print("Schema will be auto-detected from CSV headers")
schema = None

# Generate list of year-months to process
months_to_process = []
for year in range(START_YEAR, END_YEAR + 1):
    start_m = START_MONTH if year == START_YEAR else 1
    end_m = END_MONTH if year == END_YEAR else 12
    
    for month in range(start_m, end_m + 1):
        months_to_process.append(f"{year}-{month:02d}")

print(f"Processing {len(months_to_process)} months\n")
print("="*70)

# Track results
results = {
    "success": [],
    "failed": [],
    "empty": [],
    "total_rows": 0
}

# Configure load job (common for all months)
table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Process each month
for idx, year_month in enumerate(months_to_process, 1):
    print(f"\n[{idx}/{len(months_to_process)}] Processing {year_month}...")
    print("-" * 70)
    
    # Check if files exist for this month
    bucket = storage_client.bucket(BUCKET_NAME)
    month_prefix = f"{PREFIX}{year_month}/"
    blobs = list(bucket.list_blobs(prefix=month_prefix))
    csv_blobs = [blob for blob in blobs if blob.name.endswith('.csv')]
    
    if not csv_blobs:
        print(f"⚠ No CSV files found for {year_month}")
        results["empty"].append(year_month)
        continue
    
    print(f"  Found {len(csv_blobs)} CSV files")
    
    # Build GCS URI pattern for this month
    csv_pattern = f"gs://{BUCKET_NAME}/{month_prefix}*.csv"
    
    try:
        # Configure job for this month
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=True, # Changed from schema=schema
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,  # Append to existing table
            allow_jagged_rows=True,
            max_bad_records=1000,
            ignore_unknown_values=True
        )
        
        # Load data
        print(f"  Loading: {csv_pattern}")
        load_job = bq_client.load_table_from_uri(
            csv_pattern,
            table_ref,
            job_config=job_config
        )
        
        # Wait for completion
        load_job.result()
        
        # Get stats
        rows_loaded = load_job.output_rows or 0
        results["total_rows"] += rows_loaded
        results["success"].append(year_month)
        
        print(f"  ✓ Success: {rows_loaded:,} rows loaded")
        
        if load_job.errors:
            print(f"  ⚠ Warnings: {len(load_job.errors)} errors encountered (but job succeeded)")
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        results["failed"].append(year_month)

# Final Summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"✓ Successful months: {len(results['success'])}")
print(f"✗ Failed months: {len(results['failed'])}")
print(f"⚠ Empty months: {len(results['empty'])}")
print(f"\nTotal rows loaded: {results['total_rows']:,}")

if results["success"]:
    print(f"\nSuccessful: {', '.join(results['success'])}")

if results["failed"]:
    print(f"\nFailed: {', '.join(results['failed'])}")
    print("  → Retry these months individually to see detailed errors")

if results["empty"]:
    print(f"\nEmpty (no files): {', '.join(results['empty'])}")
    print("  → Run download script to fetch these months")

print("\n" + "="*70)
print(f"Table: {table_ref}")
print(f"Location: https://console.cloud.google.com/bigquery?project={PROJECT_ID}&p={PROJECT_ID}&d={DATASET_ID}&t={TABLE_ID}&page=table")
print("="*70)

# Query to verify data by month
print("\nVerify data loaded with this query:")
print(f"""
SELECT 
  EXTRACT(YEAR FROM arrival_time) as year,
  EXTRACT(MONTH FROM arrival_time) as month,
  COUNT(*) as records
FROM `{table_ref}`
GROUP BY year, month
ORDER BY year, month
""")
