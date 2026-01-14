"""
Download Historic Service Alerts
=================================
Downloads the MTA service alerts archive (~120MB) from the NY Open Data
portal and uploads to Cloud Storage.

Usage:
    python download_alerts.py

Environment Variables:
    GCP_PROJECT_ID: Google Cloud project ID
    GCP_BUCKET: Cloud Storage bucket name
    ALERTS_URL: HTTP endpoint for alerts file
"""

import os
import requests
from google.cloud import storage
from tqdm import tqdm

# Configuration - UPDATE THESE
# NY Open Data: https://data.ny.gov/Transportation/MTA-Service-Alerts-Beginning-April-2020/e53t-y7f9
ALERTS_URL = os.environ.get(
    'ALERTS_URL', 
    'https://data.ny.gov/api/views/e53t-y7f9/rows.csv?accessType=DOWNLOAD'
)
CHUNK_SIZE = 1024 * 1024  # 1MB chunks


def download_alerts(bucket_name: str, url: str):
    """
    Download service alerts CSV to GCS.
    """
    print(f"Downloading alerts from: {url}")
    print(f"Target: gs://{bucket_name}/raw/alerts/service_alerts.csv")
    
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob('raw/alerts/service_alerts.csv')
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with blob.open('wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Upload complete: gs://{bucket_name}/raw/alerts/service_alerts.csv")


def main():
    bucket_name = os.environ.get('GCP_BUCKET')
    url = os.environ.get('ALERTS_URL', ALERTS_URL)
    
    if not bucket_name:
        raise ValueError("GCP_BUCKET environment variable required")
    
    download_alerts(bucket_name, url)
    print("Done!")


if __name__ == '__main__':
    main()
