"""
Script: Download GTFS Static Files
Purpose: Downloads and extracts GTFS static data to Google Cloud Storage.

Description:
    This script downloads the GTFS static feed (subway schedule data),
    extracts the contents, and uploads to GCS under the static-files/ prefix.

Usage:
    python download_gtfs.py

Environment Variables:
    GCP_PROJECT_ID: Google Cloud Project ID.
    GCP_BUCKET: Target GCS bucket.
    GTFS_URL: URL to download GTFS zip file.
"""

import requests
from google.cloud import storage
import os
import zipfile
import tempfile
from dotenv import load_dotenv

load_dotenv()

# ============================================
# Configuration
# ============================================
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCS_BUCKET_NAME = os.environ.get("GCP_BUCKET")
GTFS_URL = os.environ.get("GTFS_URL", "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip")
GCS_PREFIX = "static-files/"

print("=" * 60)
print("Downloading GTFS Static Files")
print("=" * 60)
print(f"Source: {GTFS_URL}")
print(f"Destination: gs://{GCS_BUCKET_NAME}/{GCS_PREFIX}")
print("")

# ============================================
# Download and Extract
# ============================================
try:
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET_NAME)
    
    print("Downloading GTFS zip file...")
    response = requests.get(GTFS_URL, timeout=120, stream=True)
    
    if response.status_code != 200:
        print(f"ERROR: HTTP {response.status_code}")
        exit(1)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save zip file
        zip_path = os.path.join(temp_dir, "gtfs.zip")
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✓ Downloaded ({os.path.getsize(zip_path):,} bytes)")
        
        # Extract
        extract_dir = os.path.join(temp_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Upload to GCS
        print("Uploading to GCS...")
        uploaded_count = 0
        
        for root, _, files in os.walk(extract_dir):
            for file in files:
                local_path = os.path.join(root, file)
                # Preserve subdirectory structure if any
                relative_path = os.path.relpath(local_path, extract_dir)
                gcs_path = f"{GCS_PREFIX}{relative_path}"
                
                bucket.blob(gcs_path).upload_from_filename(local_path)
                uploaded_count += 1
                print(f"  ✓ {gcs_path}")
        
        print("")
        print("=" * 60)
        print(f"Complete: {uploaded_count} files uploaded to gs://{GCS_BUCKET_NAME}/{GCS_PREFIX}")
        print("=" * 60)

except Exception as e:
    print(f"ERROR: {e}")
    exit(1)
