
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import os
from typing import List
from src.gtfs_provider import GtfsProvider
from src.constants import (
    TARGET_COL,
    ROUTE_COL_PREFIX,
    SCHEDULED_HEADWAY_COL,
    HOUR_SIN, HOUR_COS,
    DAY_SIN, DAY_COS
)

def transform_headways(headways: np.ndarray) -> np.ndarray:
    """Apply log1p transformation to headways."""
    return np.log1p(headways)

def encode_routes(route_ids: np.ndarray, config_route_ids: List[str]) -> np.ndarray:
    """
    One-hot encode route IDs based on config.route_ids.
    Returns array of shape (n, num_routes).
    """
    # Create mapping from config
    # Ensure mapping is deterministic by sorting
    # OmegaConf ListConfig behaves like list
    sorted_routes = sorted(list(config_route_ids))
    route_mapping = {r: i for i, r in enumerate(sorted_routes)}
    num_routes = len(sorted_routes)
    
    print(f"Encoding routes using mapping: {route_mapping}")
    
    n = len(route_ids)
    onehot = np.zeros((n, num_routes))
    
    for i, route in enumerate(route_ids):
        if route in route_mapping:
            onehot[i, route_mapping[route]] = 1
        else:
            # Handle unexpected routes if necessary, or let it stay 0
            pass
            
    return onehot

def create_temporal_features(
    timestamps: np.ndarray,
    time_of_day_seconds: np.ndarray
):
    """
    Create cyclical temporal features.
    Returns: hour_sin, hour_cos, day_sin, day_cos
    """
    # Hour of day (0-23) -> radians
    # time_of_day_seconds is seconds from midnight
    hours = time_of_day_seconds / 3600
    hour_radians = 2 * np.pi * hours / 24
    hour_sin = np.sin(hour_radians)
    hour_cos = np.cos(hour_radians)
    
    # Day of week (0-6) -> radians
    timestamps_pd = pd.to_datetime(timestamps)
    day_of_week = timestamps_pd.dayofweek.values
    day_radians = 2 * np.pi * day_of_week / 7
    day_sin = np.sin(day_radians)
    day_cos = np.cos(day_radians)
    
    return hour_sin, hour_cos, day_sin, day_cos

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Pipeline overrides
    input_path = cfg.paths.input_path
    output_path = cfg.paths.output_path

    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=['arrival_time'])
    
    print(f"Data loaded. Shape: {df.shape}")
    
    # 1. Transform Targets
    print("Transforming headways (log1p)...")
    log_headway = transform_headways(df['headway'].values)
    
    # 2. Encode Categoricals
    print("One-hot encoding routes...")
    route_onehot = encode_routes(df['route_id'].values, cfg.model.route_ids)
    
    # 3. Create Cyclical Features
    print("Creating temporal features...")
    hour_sin, hour_cos, day_sin, day_cos = create_temporal_features(
        df['arrival_time'].values,
        df['time_of_day_seconds'].values
    )

    # 3.5 Generate Schedule Baseline (Metadata)
    print("Generating scheduled baseline (A32S)...")
    try:
        # Timezone Handling: Assume Input is UTC, convert to Eastern Naive
        ts_series = df['arrival_time']
        if ts_series.dt.tz is None:
            ts_series = ts_series.dt.tz_localize('UTC')
        ts_local = ts_series.dt.tz_convert('US/Eastern').dt.tz_localize(None)
        
        start_d = ts_local.min().normalize()
        end_d = ts_local.max().normalize() + pd.Timedelta(days=1)
        
        # Use DATA_LAKE_BUCKET from config
        data_bucket = cfg.pipeline.data_lake_bucket
        if not data_bucket:
             print("Warning: DATA_LAKE_BUCKET not set. Using default...")
             # fallback logic
        
        provider = GtfsProvider(gcs_bucket=data_bucket, gcs_prefix="gtfs/")
        df_sched = provider.get_scheduled_arrivals('A32S', start_d, end_d)
        
        # Merge Asof requires sorting
        # We preserve index to restore order
        temp = pd.DataFrame({'arrival_time': ts_local, 'orig_idx': df.index})
        temp_sorted = temp.sort_values('arrival_time')
        df_sched_sorted = df_sched.sort_values('timestamp')
        
        merged = pd.merge_asof(
            temp_sorted,
            df_sched_sorted[['timestamp', 'scheduled_headway_min']],
            left_on='arrival_time',
            right_on='timestamp',
            direction='backward'
        )
        
        # Restore order
        merged = merged.sort_values('orig_idx')
        # Fill missing schedule with mean (neutral baseline)
        mean_sched = merged['scheduled_headway_min'].mean()
        
        # If no schedule data matched (all NaNs), fallback to 3.0
        if pd.isna(mean_sched):
            print("Warning: No schedule overlap found. Using 3.0 min fallback.")
            mean_sched = 3.0
            
        scheduled_headway = merged['scheduled_headway_min'].fillna(mean_sched).values
        print(f"Baseline generated. Mean: {mean_sched:.2f} min")
        
    except Exception as e:
        print(f"Warning: Failed to generate baseline: {e}")
        # Fallback to mean actual headway (converted to min) approx
        print("Using fallback baseline (3.0 min)")
        scheduled_headway = np.full(len(df), 3.0)
    
    # 4. Assemble Final DataFrame
    # Structure: [log_headway, route_A, route_C, route_E, hour_sin, hour_cos, day_sin, day_cos, scheduled_headway]
    # We rely on the sorted order of cfg.model.route_ids for column naming
    sorted_routes = sorted(list(cfg.model.route_ids))
    route_columns = [f"{ROUTE_COL_PREFIX}{r}" for r in sorted_routes]
    
    # Create dictionary for DataFrame creation
    data_dict = {
        TARGET_COL: log_headway
    }
    
    # Add route columns
    for i, col in enumerate(route_columns):
        data_dict[col] = route_onehot[:, i]
        
    # Add temporal columns
    data_dict[HOUR_SIN] = hour_sin
    data_dict[HOUR_COS] = hour_cos
    data_dict[DAY_SIN] = day_sin
    data_dict[DAY_COS] = day_cos
    
    # Add Baseline Metadata
    data_dict[SCHEDULED_HEADWAY_COL] = scheduled_headway
    
    processed_df = pd.DataFrame(data_dict)
    
    print(f"Preprocessing complete. Final shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving processed data to {output_path}...")
    processed_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
