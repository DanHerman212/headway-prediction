"""
Preprocessing Module for A1 Track

Transforms raw CSV data into model-ready numpy arrays with:
- Log transformation for headway (outlier handling)
- One-hot encoding for route_id
- Cyclical encoding for temporal features (hour, day_of_week)

Input: raw_data.csv
Output: preprocessed_data.npy, scaler_params.json

Feature vector (8 features per timestep):
- log_headway (1)
- route_A, route_C, route_E (3) - one-hot
- hour_sin, hour_cos (2) - cyclical daily
- dow_sin, dow_cos (2) - cyclical weekly
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from .config import config


def load_raw_data(input_path: str = None) -> pd.DataFrame:
    """Load raw CSV data."""
    if input_path is None:
        input_path = config.raw_data_path
    
    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} records")
    return df


def log_transform_headway(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to headway values.
    
    log(headway + 1) handles outliers and reduces skewness.
    From EDA: 170x skewness reduction with log transform.
    
    Args:
        df: DataFrame with 'headway' column
    
    Returns:
        DataFrame with 'log_headway' column added
    """
    print(f"\nApplying log transformation to headway...")
    print(f"  Original range: [{df['headway'].min():.2f}, {df['headway'].max():.2f}] minutes")
    
    df['log_headway'] = np.log(df['headway'] + config.LOG_OFFSET)
    
    print(f"  Log range: [{df['log_headway'].min():.4f}, {df['log_headway'].max():.4f}]")
    print(f"  Formula: log(headway + {config.LOG_OFFSET})")
    
    # Store original headway stats for inverse transform
    headway_stats = {
        'min': float(df['headway'].min()),
        'max': float(df['headway'].max()),
        'mean': float(df['headway'].mean()),
        'std': float(df['headway'].std()),
        'log_offset': config.LOG_OFFSET
    }
    
    return df, headway_stats


def one_hot_encode_route(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode route_id into binary features.
    
    Routes: A, C, E → route_A, route_C, route_E
    
    Args:
        df: DataFrame with 'route_id' column
    
    Returns:
        DataFrame with one-hot encoded route columns
    """
    print(f"\nOne-hot encoding route_id...")
    print(f"  Routes found: {df['route_id'].unique().tolist()}")
    
    # Create one-hot encoded columns
    route_dummies = pd.get_dummies(df['route_id'], prefix='route')
    
    # Ensure all expected routes are present (even if not in data)
    for route in ['A', 'C', 'E']:
        col_name = f'route_{route}'
        if col_name not in route_dummies.columns:
            route_dummies[col_name] = 0
            print(f"  Warning: Route {route} not found in data, added as zeros")
    
    # Concatenate with original dataframe
    df = pd.concat([df, route_dummies[['route_A', 'route_C', 'route_E']]], axis=1)
    
    print(f"  Created columns: route_A, route_C, route_E")
    print(f"  Distribution:")
    for route in ['A', 'C', 'E']:
        count = df[f'route_{route}'].sum()
        pct = count / len(df) * 100
        print(f"    Route {route}: {count:,} ({pct:.1f}%)")
    
    return df


def cyclical_encode_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode temporal features using sine/cosine transformation.
    
    Captures periodicity:
    - Daily: hour_of_day (0-23) → hour_sin, hour_cos
    - Weekly: day_of_week (0-6) → dow_sin, dow_cos
    
    Args:
        df: DataFrame with 'hour_of_day' and 'day_of_week' columns
    
    Returns:
        DataFrame with cyclical temporal features
    """
    print(f"\nEncoding temporal features (cyclical)...")
    
    # Daily periodicity (24 hours)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    print(f"  Created: hour_sin, hour_cos (daily cycle)")
    
    # Weekly periodicity (7 days)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    print(f"  Created: dow_sin, dow_cos (weekly cycle)")
    
    return df


def create_feature_array(df: pd.DataFrame) -> np.ndarray:
    """
    Extract feature columns into numpy array.
    
    Feature order matches config.FEATURE_NAMES:
    [log_headway, route_A, route_C, route_E, hour_sin, hour_cos, dow_sin, dow_cos]
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        Numpy array of shape (n_samples, n_features)
    """
    print(f"\nCreating feature array...")
    
    feature_cols = config.FEATURE_NAMES
    
    # Verify all features are present
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    # Extract features in consistent order
    X = df[feature_cols].values.astype(np.float32)
    
    print(f"  Shape: {X.shape} (samples × features)")
    print(f"  Features: {feature_cols}")
    print(f"  Data type: {X.dtype}")
    
    # Check for NaN or Inf values
    if np.isnan(X).any():
        print(f"  ⚠️  Warning: {np.isnan(X).sum()} NaN values detected")
    if np.isinf(X).any():
        print(f"  ⚠️  Warning: {np.isinf(X).sum()} Inf values detected")
    
    return X


def preprocess_pipeline(input_path: str = None, output_path: str = None) -> Tuple[np.ndarray, Dict]:
    """
    Complete preprocessing pipeline.
    
    Args:
        input_path: Path to raw CSV file
        output_path: Path to save preprocessed numpy array
    
    Returns:
        Tuple of (feature_array, metadata_dict)
    """
    if output_path is None:
        output_path = config.preprocessed_data_path
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("A1 Track Preprocessing Pipeline")
    print("="*60)
    
    # Step 1: Load raw data
    df = load_raw_data(input_path)
    
    # Step 2: Log transform headway
    df, headway_stats = log_transform_headway(df)
    
    # Step 3: One-hot encode route
    df = one_hot_encode_route(df)
    
    # Step 4: Cyclical temporal encoding
    df = cyclical_encode_temporal(df)
    
    # Step 5: Create feature array
    X = create_feature_array(df)
    
    # Save preprocessed data
    print(f"\nSaving preprocessed data...")
    print(f"  Output: {output_path}")
    np.save(output_path, X)
    print(f"  ✓ Saved numpy array: {X.shape}")
    
    # Save metadata (for inverse transforms and reference)
    metadata = {
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'feature_names': config.FEATURE_NAMES,
        'headway_stats': headway_stats,
        'date_range': {
            'start': str(df['arrival_time'].min()),
            'end': str(df['arrival_time'].max())
        }
    }
    
    # Save metadata to same directory as output
    metadata_path = output_path.replace('.npy', '_metadata.json')
    print(f"  Metadata: {metadata_path}")
    
    # Create parent directory if it doesn't exist (for GCS paths)
    import os
    metadata_dir = os.path.dirname(metadata_path)
    if metadata_dir and not metadata_path.startswith('gs://'):
        os.makedirs(metadata_dir, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata JSON")
    
    return X, metadata


def main():
    """Command-line interface for preprocessing."""
    parser = argparse.ArgumentParser(description='Preprocess A1 track data')
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help=f'Input CSV path (default: {config.raw_data_path})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help=f'Output numpy path (default: {config.preprocessed_data_path})'
    )
    
    args = parser.parse_args()
    
    X, metadata = preprocess_pipeline(input_path=args.input, output_path=args.output)
    
    print()
    print("="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Samples: {metadata['n_samples']:,}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
    print(f"\nNext step: Train model")
    print(f"  python train.py")


if __name__ == "__main__":
    main()
