"""
Script: Delete Trips Files
Purpose: Cleanup utility to remove intermediate `_trips.csv` files from Google Cloud Storage.

Description:
    The ETL process extracts two types of files: `stop_times` and `trips`.
    The `trips` files contain metadata that is often redundant or not needed for the
    primary analysis, which focuses on `stop_times`. To save storage costs and keep
    the bucket clean, this script iterates through the `decompressed/` directory
    in the GCS bucket and deletes any file ending in `_trips.csv`.

Usage:
    This script is intended to be run as a task within the Airflow DAG, typically
    after the data loading phase is complete.

Environment Variables:
    GCP_PROJECT_ID: Google Cloud Project ID.
    GCS_BUCKET_NAME: Name of the GCS bucket containing the data.
"""

from google.cloud import storage
import os

import argparse
from datetime import datetime

# ============================================
# Configuration
# ============================================
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "time-series-478616")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", f"{PROJECT_ID}-historical-data")
GCS_DECOMPRESSED_PREFIX = "decompressed/"

def parse_args():
    parser = argparse.ArgumentParser(description='Delete Trips Files')
    parser.add_argument('--date', type=str, help='Specific date to clean up (YYYY-MM-DD). If omitted, scans all.')
    return parser.parse_args()

print("="*60)
print("Deleting *_trips.csv files from GCS")
print("="*60)

args = parse_args()
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(GCS_BUCKET_NAME)

# ============================================
# Find files to delete
# ============================================
trips_files = []

if args.date:
    # Targeted cleanup for a specific date
    try:
        dt = datetime.strptime(args.date, '%Y-%m-%d')
        year_month = dt.strftime('%Y-%m')
        date_str = args.date
        
        # Target specific month folder
        prefix = f"{GCS_DECOMPRESSED_PREFIX}{year_month}/"
        print(f"Scanning {prefix} for {date_str} *_trips.csv files...\n")
        
        for blob in bucket.list_blobs(prefix=prefix):
            # Match specific date pattern in filename (e.g., subwaydatanyc_2025-12-28_trips.csv)
            if date_str in blob.name and blob.name.endswith('_trips.csv'):
                trips_files.append(blob)
                
    except ValueError:
        print(f"Error: Date format must be YYYY-MM-DD. Got {args.date}")
        exit(1)
else:
    # Full scan (Backfill mode)
    print(f"Scanning ALL of {GCS_DECOMPRESSED_PREFIX} for *_trips.csv files...\n")
    for blob in bucket.list_blobs(prefix=GCS_DECOMPRESSED_PREFIX):
        if blob.name.endswith('_trips.csv'):
            trips_files.append(blob)

print(f"Found {len(trips_files)} *_trips.csv files\n")

# ============================================
# Confirm before deletion
# ============================================
if len(trips_files) == 0:
    print("No *_trips.csv files found to delete")
else:
    print("Sample files to be deleted:")
    for blob in trips_files[:5]:
        print(f"  - {blob.name}")
    if len(trips_files) > 5:
        print(f"  ... and {len(trips_files) - 5} more")
    
    print("\n" + "-"*60)
    
    # Check for FORCE_DELETE env var
    force_delete = os.environ.get("FORCE_DELETE", "false").lower() == "true"
    
    if force_delete:
        response = 'yes'
        print("FORCE_DELETE=true: Proceeding with deletion.")
    else:
        response = input(f"Delete {len(trips_files)} files? (yes/no): ")
    
    if response.lower() == 'yes':
        print("\n" + "="*60)
        print("Deleting files...")
        print("="*60 + "\n")
        
        deleted_count = 0
        error_count = 0
        
        print(f"Deleting {len(trips_files)} *_trips.csv files...")
        for blob in trips_files:
            try:
                blob.delete()
                deleted_count += 1
                if deleted_count % 100 == 0:
                    print(f"  Deleted {deleted_count} files...")
            except Exception as e:
                error_count += 1
                print(f"  ✗ Error deleting {blob.name}: {e}")
        
        print("\n" + "="*60)
        print(f"Deletion complete: {deleted_count}/{len(trips_files)} files deleted")
        if error_count > 0:
            print(f"Errors: {error_count} files failed to delete")
        print("="*60)
    else:
        print("\nDeletion cancelled")
