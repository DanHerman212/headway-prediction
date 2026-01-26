
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

# --- Configuration ---
START_DATE = '2025-07-18'
END_DATE = '2026-01-19'
TARGET_STOP_ID = 'A32S'
DEBUG = True

print("Loading GTFS data...")
trips = pd.read_csv('data/gtfs_static/trips.txt')
stop_times = pd.read_csv('data/gtfs_static/stop_times.txt')
calendar = pd.read_csv('data/gtfs_static/calendar.txt', parse_dates=['start_date', 'end_date'])
calendar_dates = pd.read_csv('data/gtfs_static/calendar_dates.txt', parse_dates=['date'])

print("Filtering Data...")
# Filter stop_times for our stop
st = stop_times[stop_times['stop_id'] == TARGET_STOP_ID].copy()
# Join with trips to get route_id and service_id
st = pd.merge(st, trips[['trip_id', 'route_id', 'service_id']], on='trip_id')

def gtfs_time_to_seconds(t_str):
    parts = t_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

st['arrival_seconds'] = st['arrival_time'].apply(gtfs_time_to_seconds)
st['hour_of_day'] = (st['arrival_seconds'] // 3600) % 24

# --- Generate List of Dates ---
date_range = pd.date_range(start=START_DATE, end=END_DATE)
schedule_rows = []

print(f"Generating schedule for {len(date_range)} days...")

# Calendar dict: service_id -> (start, end, days_mask)
# days_mask: [mon, tue, wed, thu, fri, sat, sun]
cal_map = {}
for _, row in calendar.iterrows():
    mask = [row['monday'], row['tuesday'], row['wednesday'], row['thursday'], 
            row['friday'], row['saturday'], row['sunday']]
    cal_map[row['service_id']] = (row['start_date'], row['end_date'], mask)

# Exception dict: (date, service_id) -> exception_type (1=add, 2=remove)
exc_map = {}
for _, row in calendar_dates.iterrows():
    exc_map[(row['date'], row['service_id'])] = row['exception_type']

for d in tqdm(date_range):
    day_of_week = d.dayofweek # 0=Mon, 6=Sun
    valid_services = set()
    
    # 1. Check regular calendar
    for sid, (start, end, mask) in cal_map.items():
        if start <= d <= end and mask[day_of_week] == 1:
            valid_services.add(sid)
            
    # 2. Apply exceptions
    # Additions
    # We need to find service_ids added for this date
    # Inefficient to scan full list every day, but acceptable for dataset size
    # Optimizing: pre-filter exceptions by date?
    # Let's iterate services known in `cal_map` or `exc_map`?
    # Simpler: Iterate all unique service_ids in data?
    
    # Let's use the exc_map directly
    # Check "removals"
    to_remove = set()
    for sid in valid_services:
        if exc_map.get((d, sid)) == 2:
            to_remove.add(sid)
    valid_services -= to_remove
    
    # Check "additions"
    # Find all keys in exc_map with this date and type 1
    # Optimization: Filter calendar_dates outside loop?
    # Doing it crudely here for robustness
    daily_excs = calendar_dates[calendar_dates['date'] == d]
    adds = daily_excs[daily_excs['exception_type'] == 1]['service_id'].unique()
    valid_services.update(adds)
    
    if not valid_services:
        continue
        
    # Get trips for these services from our pre-filtered stop data
    daily_trips = st[st['service_id'].isin(valid_services)].copy()
    
    if daily_trips.empty:
        continue

    # --- APPLY HEURISTIC FILTER ---
    # Day (05:00 - 21:59): C, E only.
    # Night (22:00 - 04:59): A, E only.
    # Logic:
    # If hour >= 5 and hour < 22: Keep C, E
    # Else: Keep A, E (and C if it exists? C stops running, but if it exists let's keep it? 
    # User said: "after 10pm A train ... runs local". "C and E ... during 5am to 10pm".
    # Safest interpretation:
    # 05-22: {'C', 'E'}
    # 22-05: {'A', 'E', 'C'} (Keep C if it finishes a run)
    
    mask_day = (daily_trips['hour_of_day'] >= 5) & (daily_trips['hour_of_day'] < 22)
    mask_night = ~mask_day
    
    # Filter Routes
    # Day: Keep C, E
    day_trips = daily_trips[mask_day & daily_trips['route_id'].isin(['C', 'E'])]
    
    # Night: Keep A, E, C (allow A)
    night_trips = daily_trips[mask_night & daily_trips['route_id'].isin(['A', 'C', 'E'])]
    
    valid_daily = pd.concat([day_trips, night_trips])
    
    if valid_daily.empty:
        continue
        
    # Calculate timestamps
    # GTFS seconds are from noon minus 12h? No, from midnight.
    # But trips > 24h belong to the service date effectively.
    # We construct actual timestamp: Date + Seconds
    # Note: 25:00:00 means 1am next day.
    
    # Vectorized timestamp creation
    # Base timestamp is d (00:00:00)
    # Plus timedelta
    
    # Problem: pd.to_timedelta supports days, seconds.
    # valid_daily['arrival_seconds'] is int.
    # We can use pd.to_timedelta(valid_daily['arrival_seconds'], unit='s') + d
    
    timestamps = d + pd.to_timedelta(valid_daily['arrival_seconds'], unit='s')
    
    # Prepare rows
    # We need: timestamp, route_id
    current_df = pd.DataFrame({
        'scheduled_time': timestamps,
        'route_id': valid_daily['route_id']
    })
    
    schedule_rows.append(current_df)

print("Concatenating full schedule...")
full_schedule = pd.concat(schedule_rows)
full_schedule = full_schedule.sort_values('scheduled_time')

# Calculate Headways
print("Calculating Scheduled Headways...")
full_schedule['prev_time'] = full_schedule['scheduled_time'].shift(1)
full_schedule['sched_headway'] = (full_schedule['scheduled_time'] - full_schedule['prev_time']).dt.total_seconds()

# Clean up (first row has NaN)
full_schedule = full_schedule.dropna().reset_index(drop=True)

# Select relevant columns
final_df = full_schedule[['scheduled_time', 'sched_headway']]

output_path = 'data/baseline_schedule_full.csv'
final_df.to_csv(output_path, index=False)
print(f"Saved baseline schedule to {output_path}")
print(final_df.head())
print(final_df.describe())
