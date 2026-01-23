import pandas as pd
import numpy as np
import argparse
import os

def preprocess_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    
    # Load Data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} not found. Please ensure the BigQuery extraction step has completed successfully.")
        
    df = pd.read_csv(input_path)
    
    # Validate expected schema from BigQuery
    required_columns = ['arrival_date', 'duration', 'mbt', 'dow']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}. Expected columns: {required_columns}")

    # Ensure datetime
    if 'arrival_date' in df.columns:
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        
        # --- INSERTED PREPROCESSING SNIPPETS ---

        # duration
        # 1. Remove Bad Data: Filter out negative duration instances
        initial_count = len(df)
        df = df[df['duration'] >= 0].copy()
        print(f"Removed {initial_count - len(df)} rows with negative duration.")

        # 2. Remove Extreme Outliers: Filter out duration > 150 minutes
        # there are 30 instances of duration between 100 - 150 minutes from 2024 - 2025
        upper_duration_bound = 55
        initial_count = len(df)
        df = df[df['duration'] <= upper_duration_bound].copy()
        print(f"Removed {initial_count - len(df)} rows where duration > {upper_duration_bound} mins.")

        # 3. Analyze distribution of outliers for duration
        print("\n--- Duration Distribution (Outliers) ---")
        # Display quantiles to see the spread, including extremes
        print(df['duration'].quantile([0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]))

        # 3. Remove Extreme Outliers: Filter out mbt > 35 minutes
        # User identified that > 35 mins is likely error/garbage data (approx 4% of data)
        upper_bound = 35
        initial_count = len(df)
        df = df[df['mbt'] <= upper_bound].copy()
        print(f"Removed {initial_count - len(df)} rows where mbt > {upper_bound} mins.")

        # Analyze distribution for mbt AFTER cleaning
        cols_to_check = ['mbt']

        print("\n--- MBT Distribution (After Cleaning) ---")
        for col in cols_to_check:
            print(f"\n{col} Quantiles:")
            print(df[col].quantile([0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]))

        # --- WEATHER DATA INTEGRATION (REWRITTEN) ---
        
        # 1. Load the weather data
        # Assuming weather_data.csv is in the current working directory (copied by Dockerfile)
        if os.path.exists('weather_data.csv'):
            weather_df = pd.read_csv('weather_data.csv') 
            
            # 2. Select only the physics-based features
            weather_features = ['datetime', 'temp', 'precip', 'snow', 'snowdepth', 'visibility', 'windspeed']
            # Ensure columns exist in weather_df
            available_weather_features = [col for col in weather_features if col in weather_df.columns]
            weather_df = weather_df[available_weather_features].copy()
            
            if 'datetime' in weather_df.columns:
                # 3. Convert weather timestamp to datetime objects
                weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
                
                # 4. FIX: Align Date Ranges
                # Since weather data starts in 2024, we must drop older train data to avoid NaNs.
                print(f"Original Train Count: {len(df)}")
                df = df[df['arrival_date'] >= '2024-01-01'].copy()
                print(f"Filtered Train Count (2024+): {len(df)}")
                
                # 5. Create a "Join Key" in your Train Data
                # Round train arrival time to the nearest hour to match weather data
                df['weather_join_key'] = df['arrival_date'].dt.round('h')
                
                # Ensure join key is timezone-naive to match weather data (BigQuery is UTC, CSV is naive)
                if df['weather_join_key'].dt.tz is not None:
                    df['weather_join_key'] = df['weather_join_key'].dt.tz_localize(None)
                
                # 6. Merge
                # Left join ensures we keep all train trips
                df = pd.merge(df, weather_df, left_on='weather_join_key', right_on='datetime', how='left')
                
                # 7. Clean up
                # Forward fill missing weather data (small gaps)
                weather_cols = [col for col in ['temp', 'precip', 'snow', 'snowdepth', 'visibility', 'windspeed'] if col in df.columns]
                df[weather_cols] = df[weather_cols].ffill()
                
                # Drop the helper columns (join key and the duplicate weather timestamp)
                df.drop(columns=['weather_join_key', 'datetime'], inplace=True)
                
                print("Merge Complete. New Shape:", df.shape)
                print(df.head())
        else:
            print("Warning: weather_data.csv not found. Skipping weather integration.")

        # Rename for NeuralForecast compatibility (Universal standard here)
        # Target is MBT (y), Duration is a feature
        if 'mbt' in df.columns:
            df = df.rename(columns={'arrival_date': 'ds', 'mbt': 'y'})
        elif 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])

    # --- Feature Engineering (Universal) ---
    print("Generating features...")
    
    # 1. Rolling Features
    # Calculate rolling std with window 50
    df['rolling_std_50'] = df['y'].rolling(window=50).std().bfill()
    # Add Rolling Max to capture recent spikes (delays)
    df['rolling_max_10'] = df['y'].rolling(window=10).max().bfill()
    # Add Rolling Mean/Std 10 (mentioned in prompt text of NHITS notebook)
    df['rolling_mean_10'] = df['y'].rolling(window=10).mean().bfill()
    df['rolling_std_10'] = df['y'].rolling(window=10).std().bfill()
    df['rolling_mean_50'] = df['y'].rolling(window=50).mean().bfill()
    
    # Ensure 'duration' is present (fill with 0 or mean if missing, though it should be there)
    if 'duration' not in df.columns:
        print("Warning: 'duration' column missing. Filling with 0.")
        df['duration'] = 0.0
    else:
        df['duration'] = df['duration'].fillna(0.0)

    # 2. Cyclic Features (Commented out as per user request)
    # Weekly
    # minutes_in_week = 7 * 24 * 60
    # df['week_sin'] = np.sin(2 * np.pi * df['ds'].dt.dayofweek * 24 * 60 / minutes_in_week)
    # df['week_cos'] = np.cos(2 * np.pi * df['ds'].dt.dayofweek * 24 * 60 / minutes_in_week)

    # Daily
    # minutes_in_day = 24 * 60
    # current_minute = df['ds'].dt.hour * 60 + df['ds'].dt.minute
    # df['day_sin'] = np.sin(2 * np.pi * current_minute / minutes_in_day)
    # df['day_cos'] = np.cos(2 * np.pi * current_minute / minutes_in_day)

    # 3. Day of Week (0=Weekday, 1=Weekend)
    # dayofweek: 0=Monday, 6=Sunday. So >=5 is weekend.
    df['dow'] = (df['ds'].dt.dayofweek >= 5).astype(int)

    # Fill NaNs
    df = df.bfill()

    print(f"Final columns: {df.columns.tolist()}")
    # print(f"Saving processed data to {output_path}...")
    # # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to raw input CSV')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save processed CSV')
    args = parser.parse_args()

    preprocess_data(args.input_csv, args.output_csv)
