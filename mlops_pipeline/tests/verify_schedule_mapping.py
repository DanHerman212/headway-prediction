
import os
import pandas as pd
import logging
from src.gtfs_provider import GtfsProvider
from src.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_schedule_mapping():
    print("--- VERIFYING SCHEDULE MAPPING ACCURACY ---")
    
    # 1. Setup Provider
    # Use the bucket from config or environment
    bucket_name = os.environ.get("GCP_BUCKET", config.bucket_name)
    print(f"Using Bucket: {bucket_name}")
    
    provider = GtfsProvider(gcs_bucket=bucket_name, gcs_prefix="gtfs/")
    provider.load_files()
    
    # 2. Define Test Dates Covering Different Service Patterns
    test_dates = {
        "Weekday (Wed)": "2025-12-17",
        "Saturday":      "2025-12-20",
        "Sunday":        "2025-12-21",
        "Christmas (Thu)": "2025-12-25" 
    }
    
    stop_id = "A32S" # 59 St - Columbus Circle (A/C/B/D)
    
    for label, date_str in test_dates.items():
        print(f"\nAnalyzing: {label} [{date_str}]")
        dt = pd.Timestamp(date_str)
        
        # Get strictly one day schedule
        df_sched = provider.get_scheduled_arrivals(
            stop_id=stop_id,
            start_date=dt,
            end_date=dt
        )
        
        if df_sched.empty:
            print("  ❌ No scheduled trips found!")
            continue
            
        count = len(df_sched)
        mean_hw = df_sched['scheduled_headway_min'].mean()
        
        print(f"  Trips: {count}")
        print(f"  Mean Headway: {mean_hw:.2f} min")
        
        # Check Peak Service (8 AM - 9 AM)
        # Filter for trips between 08:00 and 09:00
        am_trips = df_sched[
            (df_sched['timestamp'].dt.hour == 8)
        ]
        am_count = len(am_trips)
        am_headway = 60.0 / am_count if am_count > 0 else 0
        
        print(f"  8AM-9AM Trips: {am_count} (Approx {am_headway:.1f} min headway)")
        
        # Check active service IDs underlying this
        # Note: get_scheduled_arrivals merges resulting times, 
        # so we check the internal logic by inferring from trip density.
        
        # Output interpretation
        if label.startswith("Weekday"):
            if count > 400: print("  ✅ Looks like a Full Weekday Schedule.")
            else: print("  ⚠️ Warning: Trip count low for a weekday.")
            
        elif label == "Saturday":
            if 300 < count < 450: print("  ✅ Looks like a Saturday Schedule.")
            elif count > 450: print("  ⚠️ Warning: Looks like a Weekday schedule (too many trips).")
            else: print("  ⚠️ Warning: Trip count low.")
            
        elif label == "Sunday":
            if count < 350: print("  ✅ Looks like a Sunday Schedule.")
            else: print("  ⚠️ Warning: Trip count high for Sunday.")

        elif label.startswith("Christmas"):
            # Christmas usually runs on Sunday schedule
            if count < 350: print("  ✅ Christmas running on Reduced/Sunday Schedule.")
            else: print("  ⚠️ Warning: Christmas running on Full Weekday Schedule? (Check GTFS calendar_dates).")

    print("\n--- DONE ---")

if __name__ == "__main__":
    verify_schedule_mapping()
