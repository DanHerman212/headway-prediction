
import pandas as pd
import numpy as np

# Load data
print("Loading GTFS data...")
trips = pd.read_csv('data/gtfs_static/trips.txt')
stop_times = pd.read_csv('data/gtfs_static/stop_times.txt')
calendar = pd.read_csv('data/gtfs_static/calendar.txt', parse_dates=['start_date', 'end_date'])
calendar_dates = pd.read_csv('data/gtfs_static/calendar_dates.txt', parse_dates=['date'])

target_stop_id = 'A32S'

print(f"Filtering for stop_id: {target_stop_id}")
# Filter stop_times for target stop
st_filtered = stop_times[stop_times['stop_id'] == target_stop_id].copy()

# Join with trips to get route info
df = pd.merge(st_filtered, trips[['trip_id', 'route_id', 'service_id']], on='trip_id')

print("Sample of raw joined data:")
print(df.head())
print(f"\nUnique Routes at {target_stop_id}: {df['route_id'].unique()}")

# Helper to convert GTFS time (HH:MM:SS) to seconds
# Note: GTFS times can be > 24:00:00
def gtfs_time_to_seconds(t_str):
    parts = t_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

df['arrival_seconds'] = df['arrival_time'].apply(gtfs_time_to_seconds)

# Analyze Route distribution by hour
# Heuristic check: C/E only? A only at night?
df['hour'] = (df['arrival_seconds'] // 3600) % 24
print("\nRoute Count by Hour (Integer):")
print(df.groupby(['hour', 'route_id']).size().unstack().fillna(0))

# We need to map 'service_id' to actual dates to generate the full schedule
# This involves expanding the calendar and calendar_dates
