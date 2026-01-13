"""
Download Static GTFS Files
===========================
Downloads the GTFS static data ZIP from MTA, extracts it,
and uploads individual files to Cloud Storage.

Usage:
    python download_gtfs.py

Environment Variables:
    GCP_PROJECT_ID: Google Cloud project ID
    GCP_BUCKET: Cloud Storage bucket name
"""

import os
import io
import zipfile
import requests
from google.cloud import storage

# MTA GTFS Static Feed URL
GTFS_URL = os.environ.get(
    'GTFS_URL',
    'https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip'
)

# Files we need from the GTFS bundle
GTFS_FILES = [
    'stops.txt',
    'stop_times.txt', 
    'routes.txt',
    'trips.txt',
    'calendar.txt',
    'calendar_dates.txt',
    'shapes.txt',
    'transfers.txt',
    'agency.txt'
]


def download_gtfs(project_id: str, bucket_name: str, url: str):
    """
    Download GTFS ZIP, extract, and upload individual files to GCS.
    """
    print(f"Downloading GTFS from: {url}")
    
    # Download ZIP to memory
    response = requests.get(url)
    response.raise_for_status()
    
    print(f"Downloaded {len(response.content) / 1024 / 1024:.1f} MB")
    
    # Initialize GCS client with explicit project
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    # Extract and upload each file
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for filename in zf.namelist():
            if filename in GTFS_FILES:
                print(f"  Uploading {filename}...")
                
                content = zf.read(filename)
                blob = bucket.blob(f'raw/gtfs/{filename}')
                blob.upload_from_string(content, content_type='text/csv')
                
                print(f"    -> gs://{bucket_name}/raw/gtfs/{filename}")
    
    print("GTFS upload complete!")


def main():
    project_id = os.environ.get('GCP_PROJECT_ID')
    bucket_name = os.environ.get('GCP_BUCKET')
    url = os.environ.get('GTFS_URL', GTFS_URL)
    
    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable required")
    if not bucket_name:
        raise ValueError("GCP_BUCKET environment variable required")
    
    download_gtfs(project_id, bucket_name, url)
    print("Done!")


if __name__ == '__main__':
    main()
