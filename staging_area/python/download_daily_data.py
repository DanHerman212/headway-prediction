"""
Script: Download Daily Data
Purpose: Fetches, decompresses, and uploads a single day's MTA subway data to Google Cloud Storage.

Description:
    This script is the "Extract" part of the Daily ETL pipeline. It performs the following:
    1.  Accepts a specific date (YYYY-MM-DD).
    2.  Downloads the compressed `.tar.xz` file for that day from `https://subwaydata.nyc/data`.
    3.  Decompresses the file.
    4.  Extracts the CSV files contained within.
    5.  Uploads the extracted CSVs to a GCS bucket under a `decompressed/YYYY-MM/` prefix.

Usage:
    python3 download_daily_data.py --date 2025-12-28

Environment Variables:
    GCP_PROJECT_ID: Google Cloud Project ID.
    GCS_BUCKET_NAME: Target GCS bucket for the data.
"""

import requests
from google.cloud import storage
from datetime import datetime
import os
import tarfile
import tempfile
import argparse
import sys

# ============================================
# Configuration
# ============================================
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCS_DECOMPRESSED_PREFIX = "decompressed/"  # Final folder for CSV files
BASE_URL = "https://subwaydata.nyc/data"

def parse_args():
    parser = argparse.ArgumentParser(description='Download Daily MTA Data')
    parser.add_argument('--date', type=str, required=True, help='Date to process (YYYY-MM-DD)')
    return parser.parse_args()

def process_date(date_str):
    """
    Downloads, extracts, and uploads data for a specific date.
    """
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Date format must be YYYY-MM-DD. Got {date_str}")
        sys.exit(1)

    year_month = date_str[:7]
    file_name = f"subwaydatanyc_{date_str}_csv.tar.xz"
    url = f"{BASE_URL}/{file_name}"
    
    print(f"Starting processing for {date_str}")
    print(f"Target URL: {url}")

    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        # Check if data already exists for this month/date to avoid re-processing
        decompressed_prefix = f"{GCS_DECOMPRESSED_PREFIX}{year_month}/"
        
        # Check if specific files for this date exist
        blobs = bucket.list_blobs(prefix=decompressed_prefix)
        if any(date_str in blob.name for blob in blobs):
            print(f"SKIPPED: CSVs for {date_str} already exist in {decompressed_prefix}")
            return

        # Download from source
        print(f"Downloading {file_name}...")
        response = requests.get(url, timeout=120, stream=True)
        
        if response.status_code == 404:
            print(f"NOT_FOUND: No data available for {date_str} (HTTP 404)")
            # Raise exception to fail the task so Airflow can retry or alert
            raise Exception(f"Data not found for {date_str}")
        elif response.status_code != 200:
            raise Exception(f"HTTP Error {response.status_code} downloading {url}")
            
        # Process file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save compressed file temporarily
            local_tar_path = os.path.join(temp_dir, file_name)
            with open(local_tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("Extracting files...")
            # Extract (tarfile handles .tar.xz natively)
            extract_dir = os.path.join(temp_dir, 'extracted')
            with tarfile.open(local_tar_path, "r:xz") as tar:
                tar.extractall(path=extract_dir)
            
            # Upload CSVs
            print("Uploading CSVs to GCS...")
            csv_count = 0
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.endswith('.csv'):
                        local_csv_path = os.path.join(root, file)
                        gcs_path = f"{GCS_DECOMPRESSED_PREFIX}{year_month}/{file}"
                        bucket.blob(gcs_path).upload_from_filename(local_csv_path)
                        print(f"Uploaded: {file}")
                        csv_count += 1
            
            print(f"SUCCESS: Uploaded {csv_count} CSVs for {date_str}")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    process_date(args.date)
