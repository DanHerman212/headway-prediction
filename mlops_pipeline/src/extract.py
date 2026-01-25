
import argparse
import os
from google.cloud import bigquery
import pandas as pd
from src.config import config

def extract_data(output_path: str):
    """
    Extracts headway data from BigQuery using configuration from .env.
    Saves the result to output_path.
    """
    print(f"Initializing BigQuery client for project: {config.project_id}")
    client = bigquery.Client(project=config.project_id)

    # Construct the query dynamically based on config
    # The table name in .env is fully qualified or dataset.table?
    # Config default is "headway_prediction.ml" (dataset.table).
    # We need to prepend project_id if it's not in the string, or trust the string.
    # The archive code used: FROM `{self.project_id}.headway_prediction.ml`
    
    table_ref = f"{config.project_id}.{config.bq_table_name}"
    
    # Format route_ids for SQL IN clause: 'A', 'C', 'E'
    route_ids_str = ", ".join([f"'{r}'" for r in config.route_ids])
    
    query = f"""
    SELECT
        arrival_time,
        route_id,
        ROUND(headway, 2) AS headway,
        time_of_day_seconds
    FROM `{table_ref}`
    WHERE track = 'S'
        AND route_id IN ({route_ids_str})
    ORDER BY arrival_time
    """
    
    print(f"Executing query on {table_ref}...")
    # print(query) # Uncomment for debugging
    
    query_job = client.query(query)
    df = query_job.to_dataframe()
    
    print(f"Data extracted. Shape before dropna: {df.shape}")
    
    # Drop rows with null headway (usually the first train of the day/sequence)
    df = df.dropna()
    print(f"Shape after dropna: {df.shape}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving raw data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Extraction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # KFP passes artifacts via command line arguments
    parser.add_argument("--output_path", type=str, required=True, help="Path to save extracted data")
    
    args = parser.parse_args()
    
    extract_data(args.output_path)
