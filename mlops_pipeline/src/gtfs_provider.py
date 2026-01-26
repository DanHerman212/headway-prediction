"""
GTFS Provider Module

Handles the retrieval and parsing of GTFS Static data to establish
baseline schedules for the ML pipeline.

Usage:
    provider = GtfsProvider(gcs_bucket="my-bucket", gcs_prefix="gtfs/")
    provider.load_files()
    df_schedule = provider.get_scheduled_arrivals(
        stop_id="A32S",
        start_date="2025-12-01",
        end_date="2025-12-31"
    )
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import storage

class GtfsProvider:
    def __init__(
        self,
        gcs_bucket: str,
        gcs_prefix: str = "gtfs/",
        local_cache_dir: str = "/tmp/gtfs_cache"
    ):
        """
        Initialize GTFS Provider.
        
        Args:
            gcs_bucket: Name of GCS bucket containing extracted GTFS files.
            gcs_prefix: Prefix/folder in bucket where files reside (e.g., 'gtfs/').
            local_cache_dir: Local directory to store downloaded files.
        """
        self.bucket_name = gcs_bucket
        self.prefix = gcs_prefix
        self.cache_dir = local_cache_dir
        self.data: dict[str, pd.DataFrame] = {}
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def _download_if_missing(self, filename: str) -> str:
        """Downloads a specific file from GCS if not present locally."""
        local_path = os.path.join(self.cache_dir, filename)
        
        if os.path.exists(local_path):
            self.logger.info(f"Using cached {filename}")
            return local_path
            
        self.logger.info(f"Downloading {filename} from gs://{self.bucket_name}/{self.prefix}")
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob_path = f"{self.prefix}{filename}"
            blob = bucket.blob(blob_path)
            blob.download_to_filename(local_path)
            return local_path
        except Exception as e:
            self.logger.error(f"Failed to download {filename}: {e}")
            raise

    def load_files(self):
        """Loads required GTFS text files into memory."""
        required_files = [
            'trips.txt', 'stop_times.txt', 'calendar.txt', 
            'calendar_dates.txt', 'stops.txt'
        ]
        
        for f in required_files:
            local_path = self._download_if_missing(f)
            
            # Parse dates for calendar files
            parse_dates = False
            if 'calendar' in f:
                parse_dates = ['start_date', 'end_date'] if f == 'calendar.txt' else ['date']
                
            self.data[f] = pd.read_csv(local_path, parse_dates=parse_dates)
            
        self.logger.info("GTFS files loaded successfully.")

    def _gtfs_time_to_seconds(self, t_str: str) -> int:
        """Converts 'HH:MM:SS' string to seconds past midnight. Handles > 24h."""
        parts = t_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    def get_scheduled_arrivals(
        self, 
        stop_id: str, 
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp,
        route_ids: list[str] = None
    ) -> pd.DataFrame:
        """
        Generates concrete scheduled arrivals for a specific stop and date range.
        
        Args:
            stop_id: GTFS Stop ID (e.g., 'A32S')
            start_date: Start of window (inclusive)
            end_date: End of window (inclusive)
            route_ids: Optional list of routes to filter (e.g., ['A', 'C', 'E'])
            
        Returns:
            DataFrame with columns: ['timestamp', 'route_id', 'scheduled_headway_min']
        """
        if not self.data:
            self.load_files()
            
        trips = self.data['trips.txt']
        stop_times = self.data['stop_times.txt']
        calendar = self.data['calendar.txt']
        calendar_dates = self.data['calendar_dates.txt']
        
        # 1. Filter stop_times for target stop
        st_filtered = stop_times[stop_times['stop_id'] == stop_id].copy()
        
        # 2. Join with trips
        df_sched = pd.merge(st_filtered, trips[['trip_id', 'route_id', 'service_id']], on='trip_id')
        
        if route_ids:
            df_sched = df_sched[df_sched['route_id'].isin(route_ids)]
            
        # 3. Convert arrival_time to seconds
        df_sched['arrival_seconds'] = df_sched['arrival_time'].apply(self._gtfs_time_to_seconds)
        
        # 4. Expand Schedule day-by-day
        schedule_rows = []
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Optimization: Pre-calculate active services for each date to avoid loop overhead
        # This is a critical logical block from the notebook
        self.logger.info(f"Expanding schedule for {stop_id} from {start_date.date()} to {end_date.date()}...")
        
        for d in date_range:
            day_of_week = d.dayofweek
            
            # Identify active services
            # A. Weekly Pattern
            mask = (calendar['start_date'] <= d) & (calendar['end_date'] >= d)
            active_weekly = calendar[mask].copy()
            
            # Filter by day of week column (monday, tuesday...)
            day_columns = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            active_services = set(active_weekly[active_weekly[day_columns[day_of_week]] == 1]['service_id'])
            
            # B. Exceptions
            exceptions = calendar_dates[calendar_dates['date'] == d]
            added = set(exceptions[exceptions['exception_type'] == 1]['service_id'])
            removed = set(exceptions[exceptions['exception_type'] == 2]['service_id'])
            
            active_services = (active_services - removed) | added
            
            if not active_services:
                continue
                
            # Get trips for these services
            daily_trips = df_sched[df_sched['service_id'].isin(active_services)].copy()
            
            if daily_trips.empty:
                continue
                
            # Construct Timestamps
            # timestamp = Date + Arrival Seconds
            daily_trips['timestamp'] = d + pd.to_timedelta(daily_trips['arrival_seconds'], unit='s')
            schedule_rows.append(daily_trips[['timestamp', 'route_id']])
            
        if not schedule_rows:
            self.logger.warning("No scheduled arrivals found for this range.")
            return pd.DataFrame(columns=['timestamp', 'route_id', 'scheduled_headway_min'])
            
        full_schedule = pd.concat(schedule_rows).sort_values('timestamp').reset_index(drop=True)
        
        # 5. Calculate Scheduled Headway
        full_schedule['prev_timestamp'] = full_schedule['timestamp'].shift(1)
        full_schedule['scheduled_headway_sec'] = (full_schedule['timestamp'] - full_schedule['prev_timestamp']).dt.total_seconds()
        full_schedule['scheduled_headway_min'] = full_schedule['scheduled_headway_sec'] / 60.0
        
        # Clean up
        # We replace the first NaN (infinity) or very large headways (overnight gaps)
        # For the purpose of baseline, we can leave overnight gaps, 
        # but in comparison we usually filter to < 60 min
        
        return full_schedule[['timestamp', 'route_id', 'scheduled_headway_min']]

