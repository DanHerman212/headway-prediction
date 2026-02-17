"""
Build the two static lookup tables for the streaming pipeline side inputs.

Reads the training parquet, filters to training period only (before TRAINING_CUTOFF),
and computes:
  1. empirical_map:  (route_id, day_type, hour) -> median service_headway
  2. median_tt_map:  (route_id, day_type, hour, station_id) -> median travel_time

Keys are stored as comma-separated strings to match the DoFn lookup format.
Output: local_artifacts/empirical_map.json, local_artifacts/median_tt_map.json
"""

import json
import pandas as pd
from datetime import datetime

TRAINING_CUTOFF = 1763424000.0
DATA_PATH = "local_artifacts/processed_data/training_data.parquet"


def main():
    print(f"Training cutoff: {datetime.fromtimestamp(TRAINING_CUTOFF).isoformat()}")

    df = pd.read_parquet(DATA_PATH)
    print(f"Full dataset: {len(df)} rows")

    at = pd.to_datetime(df["arrival_time"], utc=True)
    df["timestamp"] = at.astype("int64") // 10**9
    df["hour"] = at.dt.hour
    df["day_type"] = df["day_of_week"].apply(
        lambda d: "Weekend" if d >= 5 else "Weekday"
    )

    train = df[df["timestamp"] < TRAINING_CUTOFF].copy()
    print(f"Training subset: {len(train)} rows")

    # === 1. Empirical map ===
    emp = train.dropna(subset=["service_headway"])
    emp_groups = emp.groupby(["route_id", "day_type", "hour"])["service_headway"]
    empirical_map = {}
    for (route, day_type, hour), series in emp_groups:
        key = f"{route},{day_type},{hour}"
        empirical_map[key] = round(float(series.median()), 6)

    print(f"empirical_map: {len(empirical_map)} entries")

    # === 2. Median travel time map ===
    segments = [
        ("travel_time_14th", "A31S"),
        ("travel_time_23rd", "A30S"),
        ("travel_time_34th", "A28S"),
    ]
    median_tt_map = {}
    for tt_col, station_id in segments:
        sub = train.dropna(subset=[tt_col])
        groups = sub.groupby(["route_id", "day_type", "hour"])[tt_col]
        for (route, day_type, hour), series in groups:
            key = f"{route},{day_type},{hour},{station_id}"
            median_tt_map[key] = round(float(series.median()), 6)

    print(f"median_tt_map: {len(median_tt_map)} entries")

    # === Save ===
    with open("local_artifacts/empirical_map.json", "w") as f:
        json.dump(empirical_map, f, indent=2)
    with open("local_artifacts/median_tt_map.json", "w") as f:
        json.dump(median_tt_map, f, indent=2)

    print("Saved: local_artifacts/empirical_map.json")
    print("Saved: local_artifacts/median_tt_map.json")


if __name__ == "__main__":
    main()
