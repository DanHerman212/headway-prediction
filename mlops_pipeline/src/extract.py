
import hydra
from omegaconf import DictConfig
from google.cloud import bigquery
import pandas as pd
import os

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Extracts headway data from BigQuery using configuration from Hydra.
    """
    # Vertex AI Pipeline passes output path via override: paths.output_path=...
    output_path = cfg.paths.output_path

    print(f"Initializing BigQuery client for project: {cfg.experiment.project_id}")
    client = bigquery.Client(project=cfg.experiment.project_id)

    table_ref = f"{cfg.experiment.project_id}.{cfg.pipeline.bq_table_name}"
    
    # Format route_ids for SQL IN clause: 'A', 'C', 'E'
    # Hydra/OmegaConf lists behave like Python lists
    route_ids = cfg.model.route_ids
    route_ids_str = ", ".join([f"'{r}'" for r in route_ids])
    
    query = f"""
    SELECT
        arrival_time,
        route_id,
        ROUND(headway, 2) AS headway,
        time_of_day_seconds
    FROM `{table_ref}`
    WHERE track = 'A1'
        AND route_id IN ({route_ids_str})
    ORDER BY arrival_time
    """
    
    print(f"Executing query on {table_ref}...")
    
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
    main()
