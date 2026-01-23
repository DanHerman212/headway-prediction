import argparse
import pandas as pd
from google.cloud import bigquery
import os

def extract_data(project_id, query, output_path):
    print(f"Extracting data from BigQuery...")
    print(f"Project: {project_id}")
    print(f"Query: {query}")

    client = bigquery.Client(project=project_id)
    query_job = client.query(query)
    df = query_job.to_dataframe()
    
    print(f"Data extracted. Shape: {df.shape}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Extraction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    
    args = parser.parse_args()
    
    extract_data(args.project_id, args.query, args.output_csv)
