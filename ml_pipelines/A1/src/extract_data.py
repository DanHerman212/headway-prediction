"""
Data Extraction Module for A1 Track

Queries BigQuery for A1 track data and saves as CSV artifact.

Output CSV columns:
- arrival_time: timestamp of train arrival
- headway: time since previous train (minutes)
- route_id: train route (A, C, or E)
- track: track identifier (A1)
- time_of_day: hour of day (0-23)
- day_of_week: day of week (0=Monday, 6=Sunday)

Usage:
    python extract_data.py --output data/A1/raw_data.csv
"""

import argparse
import os
from pathlib import Path
from google.cloud import bigquery
import pandas as pd
from .config import config


def extract_a1_data(output_path: str = None) -> pd.DataFrame:
    """
    Extract A1 track data from BigQuery and save as CSV.
    
    Args:
        output_path: Path to save CSV file. If None, uses config default.
    
    Returns:
        DataFrame with raw A1 data
    """
    if output_path is None:
        output_path = config.raw_data_path
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting data from BigQuery...")
    print(f"  Project: {config.BQ_PROJECT}")
    print(f"  Dataset: {config.BQ_DATASET}")
    print(f"  Table: {config.BQ_TABLE}")
    print(f"  Filter: track = 'A1'")
    
    # Initialize BigQuery client
    client = bigquery.Client(project=config.BQ_PROJECT, location=config.BQ_LOCATION)
    
    # SQL query to extract A1 track data
    # Ordered chronologically for time series processing
    query = f"""
    SELECT 
        arrival_time,
        headway,
        route_id,
        track,
        time_of_day,
        day_of_week
    FROM `{config.BQ_PROJECT}.{config.BQ_DATASET}.{config.BQ_TABLE}`
    WHERE track = 'A1'
    ORDER BY arrival_time ASC
    """
    
    print(f"\nExecuting query...")
    df = client.query(query).to_dataframe()
    
    print(f"✓ Query complete")
    print(f"  Records: {len(df):,}")
    print(f"  Date range: {df['arrival_time'].min()} to {df['arrival_time'].max()}")
    print(f"  Routes: {df['route_id'].unique().tolist()}")
    
    # Data quality checks
    print(f"\nData Quality Checks:")
    print(f"  Missing values:")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"    {col}: {null_count} ({null_count/len(df)*100:.2f}%)")
    
    if df.isnull().sum().sum() == 0:
        print(f"    No missing values ✓")
    
    # Headway statistics
    print(f"\n  Headway statistics (minutes):")
    print(f"    Min: {df['headway'].min():.2f}")
    print(f"    Max: {df['headway'].max():.2f}")
    print(f"    Mean: {df['headway'].mean():.2f}")
    print(f"    Median: {df['headway'].median():.2f}")
    print(f"    Std: {df['headway'].std():.2f}")
    print(f"    99th percentile: {df['headway'].quantile(0.99):.2f}")
    
    # Save to CSV
    print(f"\nSaving to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"✓ CSV saved successfully")
    
    # File size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    return df


def main():
    """Command-line interface for data extraction."""
    parser = argparse.ArgumentParser(description='Extract A1 track data from BigQuery')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help=f'Output CSV path (default: {config.raw_data_path})'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("A1 Track Data Extraction")
    print("="*60)
    print()
    
    df = extract_a1_data(output_path=args.output)
    
    print()
    print("="*60)
    print("Extraction Complete!")
    print("="*60)
    print(f"\nNext step: Run preprocessing")
    print(f"  python preprocess.py --input {args.output or config.raw_data_path}")


if __name__ == "__main__":
    main()
