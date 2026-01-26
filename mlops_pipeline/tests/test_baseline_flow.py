
import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gtfs_provider import GtfsProvider
from src.preprocess import preprocess_data
from src.config import config

def test_provider_connection():
    print("\n--- Testing GtfsProvider Connection ---")
    data_bucket = os.environ.get("GCP_BUCKET", "headway-prediction-historic-data")
    print(f"Target Bucket: {data_bucket}")
    
    provider = GtfsProvider(gcs_bucket=data_bucket, gcs_prefix="gtfs/")
    try:
        # Try to download just one file to verify access
        local_path = provider._download_if_missing("trips.txt")
        print(f"Successfully downloaded trips.txt to {local_path}")
        return True
    except Exception as e:
        print(f"FAILED to download from GCS: {e}")
        return False

def test_preprocessing_integration():
    print("\n--- Testing Preprocessing Integration ---")
    
    # 1. Create Dummy Input
    input_path = "tests/test_raw_sample.csv"
    output_path = "tests/test_processed.csv"
    
    print(f"Reading input from {input_path}")
    
    # 2. Run Preprocess
    try:
        preprocess_data(input_path, output_path)
        print("Preprocessing function completed.")
    except Exception as e:
        print(f"Preprocessing FAILED: {e}")
        return

    # 3. Verify Output
    print(f"Verifying output at {output_path}")
    if not os.path.exists(output_path):
        print("Output file not found!")
        return
        
    df = pd.read_csv(output_path)
    print("Columns found:", df.columns.tolist())
    
    if 'scheduled_headway' in df.columns:
        print("SUCCESS: 'scheduled_headway' column exists.")
        print("Sample values:")
        print(df['scheduled_headway'].head())
        
        mean_val = df['scheduled_headway'].mean()
        if mean_val == 3.0 and df['scheduled_headway'].std() == 0:
             print("WARNING: All values are 3.0. Valid GCS download failed, fell back to default.")
        else:
             print(f"Values look dynamic (Mean: {mean_val:.2f})")
    else:
        print("FAILURE: 'scheduled_headway' column MISSING.")

if __name__ == "__main__":
    # Ensure env var is set for the test if not present
    if not os.environ.get("GCP_BUCKET"):
        os.environ["GCP_BUCKET"] = "headway-prediction-historic-data"
        
    # We change directory to mlops_pipeline so relative paths in test work
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if test_provider_connection():
        test_preprocessing_integration()
