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
