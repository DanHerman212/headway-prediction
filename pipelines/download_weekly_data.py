#!/usr/bin/env python3
"""
Script: Download Weekly Data
Purpose: Downloads 7 days of MTA subway data in a single batch.

Description:
    This script wraps download_daily_data.py to fetch an entire week at once.
    Intended to run every Sunday to collect the previous week's data.

Usage:
    python3 download_weekly_data.py --week-ending 2025-12-28
    
    This downloads data for Dec 22-28, 2025 (7 days ending on the given date)

Environment Variables:
    GCP_PROJECT_ID: Google Cloud Project ID.
    GCS_BUCKET_NAME: Target GCS bucket for the data.
"""

import requests
from google.cloud import storage
from datetime import datetime, timedelta
import os
import tarfile
import tempfile
import argparse
import sys
import concurrent.futures

# ============================================
# Configuration
# ============================================
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCS_DECOMPRESSED_PREFIX = "decompressed/"
BASE_URL = "https://subwaydata.nyc/data"
MAX_WORKERS = 4  # Parallel downloads


def parse_args():
    parser = argparse.ArgumentParser(description='Download Weekly MTA Data')
    parser.add_argument('--week-ending', type=str, required=True, 
                        help='End date of the week to process (YYYY-MM-DD). Downloads 7 days ending on this date.')
    return parser.parse_args()


def process_date(date_str):
    """
    Downloads, extracts, and uploads data for a specific date.
    Returns (date_str, status, message) tuple.
    """
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return (date_str, "ERROR", f"Invalid date format: {date_str}")

    year_month = date_str[:7]
    file_name = f"subwaydatanyc_{date_str}_csv.tar.xz"
    url = f"{BASE_URL}/{file_name}"

    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        # Check if data already exists
        decompressed_prefix = f"{GCS_DECOMPRESSED_PREFIX}{year_month}/"
        blobs = list(bucket.list_blobs(prefix=decompressed_prefix))
        if any(date_str in blob.name for blob in blobs):
            return (date_str, "SKIPPED", f"CSVs already exist in {decompressed_prefix}")

        # Download from source
        response = requests.get(url, timeout=120, stream=True)
        
        if response.status_code == 404:
            return (date_str, "NOT_FOUND", "No data available")
        elif response.status_code != 200:
            return (date_str, "ERROR", f"HTTP {response.status_code}")
            
        # Process file
        with tempfile.TemporaryDirectory() as temp_dir:
            local_tar_path = os.path.join(temp_dir, file_name)
            with open(local_tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            extract_dir = os.path.join(temp_dir, 'extracted')
            with tarfile.open(local_tar_path, "r:xz") as tar:
                tar.extractall(path=extract_dir)
            
            csv_count = 0
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.endswith('.csv') and '_trips.csv' not in file:
                        # Only upload stop_times, skip trips files
                        local_csv_path = os.path.join(root, file)
                        gcs_path = f"{GCS_DECOMPRESSED_PREFIX}{year_month}/{file}"
                        bucket.blob(gcs_path).upload_from_filename(local_csv_path)
                        csv_count += 1
            
            return (date_str, "SUCCESS", f"Uploaded {csv_count} CSV files")
            
    except Exception as e:
        return (date_str, "ERROR", str(e))


def main():
    args = parse_args()
    
    if not PROJECT_ID or not GCS_BUCKET_NAME:
        print("Error: GCP_PROJECT_ID and GCS_BUCKET_NAME environment variables required")
        sys.exit(1)
    
    # Parse end date
    try:
        end_date = datetime.strptime(args.week_ending, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Date format must be YYYY-MM-DD. Got {args.week_ending}")
        sys.exit(1)
    
    # Generate list of 7 dates (week ending on given date)
    dates_to_process = []
    for i in range(6, -1, -1):  # 6 days ago to today
        date = end_date - timedelta(days=i)
        dates_to_process.append(date.strftime('%Y-%m-%d'))
    
    print("=" * 60)
    print("Downloading Weekly MTA Data")
    print("=" * 60)
    print(f"Week ending: {args.week_ending}")
    print(f"Dates to process: {dates_to_process[0]} to {dates_to_process[-1]}")
    print(f"Target bucket: gs://{GCS_BUCKET_NAME}/{GCS_DECOMPRESSED_PREFIX}")
    print("=" * 60 + "\n")
    
    # Process dates in parallel
    results = {"SUCCESS": [], "SKIPPED": [], "NOT_FOUND": [], "ERROR": []}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_date, date_str): date_str for date_str in dates_to_process}
        
        for future in concurrent.futures.as_completed(futures):
            date_str, status, message = future.result()
            results[status].append(date_str)
            print(f"  [{status}] {date_str}: {message}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  SUCCESS:   {len(results['SUCCESS'])}")
    print(f"  SKIPPED:   {len(results['SKIPPED'])}")
    print(f"  NOT_FOUND: {len(results['NOT_FOUND'])}")
    print(f"  ERROR:     {len(results['ERROR'])}")
    
    if results['ERROR']:
        print(f"\nFailed dates: {results['ERROR']}")
        sys.exit(1)


if __name__ == '__main__':
    main()
