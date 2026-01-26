
import argparse
import os
import numpy as np
import pandas as pd
from src.config import config
from src.gtfs_provider import GtfsProvider

def transform_headways(headways: np.ndarray) -> np.ndarray:
    """Apply log1p transformation to headways."""
    return np.log1p(headways)

def encode_routes(route_ids: np.ndarray) -> np.ndarray:
    """
    One-hot encode route IDs based on config.route_ids.
    Returns array of shape (n, num_routes).
    """
    # Create mapping from config
    # Ensure mapping is deterministic by sorting
    sorted_routes = sorted(config.route_ids)
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

def preprocess_data(input_path: str, output_path: str):
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=['arrival_time'])
    
    print(f"Data loaded. Shape: {df.shape}")
    
    # 1. Transform Targets
    print("Transforming headways (log1p)...")
    log_headway = transform_headways(df['headway'].values)
    
    # 2. Encode Categoricals
    print("One-hot encoding routes...")
    route_onehot = encode_routes(df['route_id'].values)
    
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
        
        # Use GCP_BUCKET (Data Lake) if available, else fall back to config bucket
        data_bucket = os.environ.get("GCP_BUCKET", config.bucket_name)
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
        scheduled_headway = merged['scheduled_headway_min'].fillna(mean_sched).values
        print(f"Baseline generated. Mean: {mean_sched:.2f} min")
        
    except Exception as e:
        print(f"Warning: Failed to generate baseline: {e}")
        # Fallback to mean actual headway (converted to min) approx
        print("Using fallback baseline (3.0 min)")
        scheduled_headway = np.full(len(df), 3.0)
    
    # 4. Assemble Final DataFrame
    # Structure: [log_headway, route_A, route_C, route_E, hour_sin, hour_cos, day_sin, day_cos, scheduled_headway]
    # We rely on the sorted order of config.route_ids for column naming
    sorted_routes = sorted(config.route_ids)
    route_columns = [f"route_{r}" for r in sorted_routes]
    
    # Create dictionary for DataFrame creation
    data_dict = {
        'log_headway': log_headway
    }
    
    # Add route columns
    for i, col in enumerate(route_columns):
        data_dict[col] = route_onehot[:, i]
        
    # Add temporal columns
    data_dict['hour_sin'] = hour_sin
    data_dict['hour_cos'] = hour_cos
    data_dict['day_sin'] = day_sin
    data_dict['day_cos'] = day_cos
    
    # Add Baseline Metadata
    data_dict['scheduled_headway'] = scheduled_headway
    
    processed_df = pd.DataFrame(data_dict)
    
    print(f"Preprocessing complete. Final shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving processed data to {output_path}...")
    processed_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to raw csv")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save processed csv")
    
    args = parser.parse_args()
    
    preprocess_data(args.input_path, args.output_path)
