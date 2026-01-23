"""NYC Subway headway data extraction."""

from typing import Optional, Tuple, TYPE_CHECKING
import pandas as pd
from google.cloud import bigquery

if TYPE_CHECKING:
    from config.model_config import ModelConfig


ROUTE_MAPPING = {"A": 0, "C": 1, "E": 2}


class DataExtractor:
    """Extracts headway data from BigQuery."""
    
    def __init__(self, config: "ModelConfig"):
        """
        Initialize extractor.
        
        Args:
            config: ModelConfig instance with data extraction parameters
        """
        self.config = config
        self.project_id = config.bq_project
        self.client = bigquery.Client(project=self.project_id)
    
    def extract(self) -> pd.DataFrame:
        """
        Extract headway data from BigQuery using config parameters.
        
        Returns:
            DataFrame with columns: arrival_time, route_id, headway, time_of_day_seconds
        """
        route_ids_str = ", ".join([f"'{r}'" for r in self.config.route_ids])
        
        query = f"""
        SELECT
            arrival_time,
            route_id,
            ROUND(headway, 2) AS headway,
            time_of_day_seconds
        FROM `{self.project_id}.headway_prediction.ml`
        WHERE track = '{self.config.track}'
            AND route_id IN ({route_ids_str})
        ORDER BY arrival_time
        """
        
        df = self.client.query(query).to_dataframe()
        df = df.dropna()  # Remove null headway (first event)
        
        return df
    
    def save(self, df: pd.DataFrame, path: str) -> None:
        """Save extracted data to CSV."""
        df.to_csv(path, index=False)


if __name__ == "__main__":
    import argparse
    import logging
    import os
    import sys
    
    # Add project root to path if running as script
    # This allows imports like 'from config.model_config import ...'
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Now we can safely import config
    # Note: Using absolute imports relative to ml_pipelines package
    from ml_pipelines.config.model_config import ModelConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, required=True, help="GCP Project ID")
    parser.add_argument("--query", type=str, required=False, help="Optional custom query overrides config default")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save extracted CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Extracting data for project {args.project_id}")

    # Initialize config
    config = ModelConfig()
    config.bq_project = args.project_id
    
    # Extract
    extractor = DataExtractor(config)
    logging.info("Running extraction...")
    df = extractor.extract()
    
    logging.info(f"Data extracted: {len(df)} rows")
    
    # Save
    logging.info(f"Saving to {args.output_csv}")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    extractor.save(df, args.output_csv)
