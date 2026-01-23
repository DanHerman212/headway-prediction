"""Preprocessing for subway headway data."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, TYPE_CHECKING
from .data import ROUTE_MAPPING

if TYPE_CHECKING:
    from config.model_config import ModelConfig


class DataPreprocessor:
    """Preprocesses headway data for model training."""
    
    def __init__(self, config: "ModelConfig"):
        """
        Initialize preprocessor.
        
        Args:
            config: ModelConfig instance with preprocessing parameters
        """
        self.config = config
        self.route_mapping = ROUTE_MAPPING
        self.num_routes = len(ROUTE_MAPPING)
    
    def _transform_headways(self, headways: np.ndarray) -> np.ndarray:
        """Apply log1p transformation to headways."""
        return np.log1p(headways)
    
    def _encode_routes(self, route_ids: np.ndarray) -> np.ndarray:
        """
        One-hot encode route IDs.
        
        Args:
            route_ids: Array of route strings ('A', 'C', 'E')
            
        Returns:
            One-hot encoded array of shape (n, 3)
        """
        n = len(route_ids)
        onehot = np.zeros((n, self.num_routes))
        
        for i, route in enumerate(route_ids):
            onehot[i, self.route_mapping[route]] = 1
        
        return onehot
    
    def _create_temporal_features(
        self,
        timestamps: np.ndarray,
        time_of_day_seconds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create cyclical temporal features.
        
        Args:
            timestamps: Pandas datetime array
            time_of_day_seconds: Seconds since midnight
            
        Returns:
            hour_sin, hour_cos, day_sin, day_cos
        """
        # Hour of day (0-23) → radians
        hours = time_of_day_seconds / 3600
        hour_radians = 2 * np.pi * hours / 24
        hour_sin = np.sin(hour_radians)
        hour_cos = np.cos(hour_radians)
        
        # Day of week (0-6) → radians
        timestamps_pd = pd.to_datetime(timestamps)
        day_of_week = timestamps_pd.dayofweek.values
        day_radians = 2 * np.pi * day_of_week / 7
        day_sin = np.sin(day_radians)
        day_cos = np.cos(day_radians)
        
        return hour_sin, hour_cos, day_sin, day_cos
    
    def load(self, path: str) -> pd.DataFrame:
        """Load raw data from CSV."""
        return pd.read_csv(path, parse_dates=['arrival_time'])
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data: transform features.
        
        Args:
            df: Raw dataframe from CSV
            
        Returns:
            DataFrame with preprocessed features:
            [log_headway, route_A, route_C, route_E, hour_sin, hour_cos, day_sin, day_cos]
        """
        # Transform headways
        log_headways = self._transform_headways(df['headway'].values)
        
        # Encode routes
        route_onehot = self._encode_routes(df['route_id'].values)
        
        # Create temporal features
        hour_sin, hour_cos, day_sin, day_cos = self._create_temporal_features(
            df['arrival_time'].values,
            df['time_of_day_seconds'].values
        )
        
        # Combine
        features = np.column_stack([
            log_headways,
            route_onehot,
            hour_sin, hour_cos, day_sin, day_cos
        ])
        
        columns = [
            'log_headway',
            'route_A', 'route_C', 'route_E',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        return pd.DataFrame(features, columns=columns)


if __name__ == "__main__":
    import argparse
    import logging
    import os
    import sys
    
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from ml_pipelines.config.model_config import ModelConfig
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Preprocessing data from {args.input_csv}")

    # Initialize config and preprocessor
    config = ModelConfig()
    preprocessor = DataPreprocessor(config)

    # Load and process
    try:
        df_raw = preprocessor.load(args.input_csv)
        df_processed = preprocessor.preprocess(df_raw)

        logging.info(f"Processed data shape: {df_processed.shape}")

        # Save
        logging.info(f"Saving to {args.output_csv}")
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        df_processed.to_csv(args.output_csv, index=False)
        
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        sys.exit(1)
