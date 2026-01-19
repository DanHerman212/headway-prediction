
"""
Script: Download Historical Data
Purpose: Fetches, decompresses, and uploads MTA subway data to Google Cloud Storage.

Description:
    This script is the "Extract" part of the ETL pipeline. It performs the following:
    1.  Iterates through a specified date range (approx. last 4 years).
    2.  Downloads compressed `.tar.xz` files from `https://subwaydata.nyc/data` for each day.
    3.  Decompresses these files in memory or using temporary storage.
    4.  Extracts the CSV files contained within.
    5.  Uploads the extracted CSVs to a GCS bucket under a `decompressed/YYYY-MM/` prefix.
    6.  Uses multi-threading (concurrent.futures) to parallelize downloads and uploads for efficiency.

Usage:
    Run as a Python script or Airflow task. Requires GCS write permissions.

Environment Variables:
    GCP_PROJECT_ID: Google Cloud Project ID.
    GCS_BUCKET_NAME: Target GCS bucket for the data.
"""

import requests
from google.cloud import storage
import concurrent.futures
from datetime import datetime, timedelta
import os
import tarfile
import tempfile
import argparse
import time

# ============================================
# Configuration
# ============================================
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCS_DECOMPRESSED_PREFIX = "decompressed/"  # Final folder for CSV files
BASE_URL = "https://subwaydata.nyc/data"
MAX_WORKERS = 10  # Number of parallel downloads

def parse_args():
    parser = argparse.ArgumentParser(description='Download MTA Data')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    return parser.parse_args()

# ============================================
# Core Logic
# ============================================
def process_date(date):
    """
    Downloads, extracts, and uploads data for a specific date.
    """
    date_str = date.strftime('%Y-%m-%d')
    year_month = date_str[:7]
    file_name = f"subwaydatanyc_{date_str}_csv.tar.xz"
    url = f"{BASE_URL}/{file_name}"
    
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        # Check if data already exists for this month/date to avoid re-processing
        decompressed_prefix = f"{GCS_DECOMPRESSED_PREFIX}{year_month}/"
        blobs = bucket.list_blobs(prefix=decompressed_prefix)
        if any(date_str in blob.name for blob in blobs):
            return (date_str, "SKIPPED", f"CSVs already in {year_month}/")
        
        # Download from source
        response = requests.get(url, timeout=120, stream=True)
        
        if response.status_code == 404:
            return (date_str, "NOT_FOUND", "No data available")
        elif response.status_code != 200:
            return (date_str, "ERROR", f"HTTP {response.status_code}")
            
        # Process file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save compressed file temporarily
            local_tar_path = os.path.join(temp_dir, file_name)
            with open(local_tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract (tarfile handles .tar.xz natively)
            extract_dir = os.path.join(temp_dir, 'extracted')
            with tarfile.open(local_tar_path, "r:xz") as tar:
                tar.extractall(path=extract_dir)
            
            # Upload CSVs
            csv_count = 0
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.endswith('.csv'):
                        local_csv_path = os.path.join(root, file)
                        gcs_path = f"{GCS_DECOMPRESSED_PREFIX}{year_month}/{file}"
                        bucket.blob(gcs_path).upload_from_filename(local_csv_path)
                        csv_count += 1
            
            return (date_str, "SUCCESS", f"Uploaded {csv_count} CSVs")

    except Exception as e:
        return (date_str, "ERROR", str(e))

# ============================================
# Main Execution
# ============================================
def main():
    args = parse_args()
    
    # Determine date range
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        print("No dates provided. Defaulting to historical backfill range.")
        DAYS = 1729 
        end_date = datetime(2025, 12, 25)
        start_date = end_date - timedelta(days=DAYS)

    print(f"\nStarting Download: {start_date.date()} to {end_date.date()}")
    
    # Ensure Bucket Exists
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        if not bucket.exists():
            bucket.create(location="US")
    except Exception as e:
        print(f"Bucket Error: {e}")
        return

    dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    results = {"SUCCESS": 0, "SKIPPED": 0, "NOT_FOUND": 0, "ERROR": 0}
    
    print(f"Processing {len(dates)} dates with {MAX_WORKERS} workers...\n")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_date = {executor.submit(process_date, date): date for date in dates}
        
        for future in concurrent.futures.as_completed(future_to_date):
            date_str, status, message = future.result()
            results[status] = results.get(status, 0) + 1
            
            symbol = {"SUCCESS": "✓", "SKIPPED": "⊘", "NOT_FOUND": "✗", "ERROR": "!"}.get(status, "?")
            print(f"{symbol} {date_str}: {status} - {message}")

    print("\n" + "="*30)
    print("Summary")
    print("="*30)
    for k, v in results.items():
        print(f"{k}: {v}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
