"""
Download Historic Subway Schedules
===================================
Downloads the large (~9GB) historic subway schedules CSV from the MTA
public HTTP endpoint and uploads to Cloud Storage.

Usage:
    python download_schedules.py

Environment Variables:
    GCP_PROJECT_ID: Google Cloud project ID
    GCP_BUCKET: Cloud Storage bucket name
    SCHEDULES_URL: HTTP endpoint for schedules file
"""

import os
import requests
from google.cloud import storage
from tqdm import tqdm

# Configuration - UPDATE THESE
SCHEDULES_URL = os.environ.get('SCHEDULES_URL', 'https://example.com/schedules.csv')
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for streaming


def download_schedules(bucket_name: str, url: str):
    """
    Stream download large schedules file directly to GCS.
    
    Uses streaming to handle the ~9GB file without loading into memory.
    """
    print(f"Downloading schedules from: {url}")
    print(f"Target: gs://{bucket_name}/raw/schedules/historic_schedules.csv")
    
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob('raw/schedules/historic_schedules.csv')
    
    # Stream download and upload
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with blob.open('wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Upload complete: gs://{bucket_name}/raw/schedules/historic_schedules.csv")


def main():
    bucket_name = os.environ.get('GCP_BUCKET')
    url = os.environ.get('SCHEDULES_URL', SCHEDULES_URL)
    
    if not bucket_name:
        raise ValueError("GCP_BUCKET environment variable required")
    
    if 'example.com' in url:
        raise ValueError("Please set SCHEDULES_URL environment variable")
    
    download_schedules(bucket_name, url)
    print("Done!")


if __name__ == '__main__':
    main()
