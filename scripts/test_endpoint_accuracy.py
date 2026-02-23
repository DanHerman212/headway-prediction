#!/usr/bin/env python3
"""
test_endpoint_accuracy.py
-------------------------
Send 30 historical test windows (10 per group) to the live Vertex AI
endpoint and report prediction accuracy against known actuals.

Groups: A_South, C_South, E_South

Usage:
  python scripts/test_endpoint_accuracy.py
"""

import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = "realtime-headway-prediction"
LOCATION = "us-east1"
ENDPOINT_DISPLAY_NAME = "headway-prediction-endpoint"

GROUP_IDS = ["A_South", "C_South", "E_South"]
WINDOW_SIZE = 20
SAMPLES_PER_GROUP = 10

FEATURES = [
    "service_headway", "preceding_train_gap", "upstream_headway_14th",
    "travel_time_14th", "travel_time_14th_deviation",
    "travel_time_23rd", "travel_time_23rd_deviation",
    "travel_time_34th", "travel_time_34th_deviation",
    "stops_at_23rd", "hour_sin", "hour_cos", "day_of_week",
    "time_idx", "empirical_median",
    "route_id", "regime_id", "track_id", "preceding_route_id",
]


def load_test_data():
    """Load training parquet, clean features, return test split."""
    path = os.path.join(os.path.dirname(__file__), "..", "local_artifacts",
                        "processed_data", "training_data.parquet")
    df = pd.read_parquet(path)

    # Categoricals
    for col in ["group_id", "route_id", "direction", "regime_id", "track_id", "preceding_route_id"]:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    # Time index
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    min_time = df["arrival_time"].min()
    df["time_idx"] = ((df["arrival_time"] - min_time).dt.total_seconds() / 60).astype(int)

    # Impute numerics
    for col in ["preceding_train_gap", "upstream_headway_14th", "travel_time_14th",
                "travel_time_34th", "travel_time_23rd"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if "preceding_train_gap" in df.columns:
        df["preceding_train_gap"] = df["preceding_train_gap"].clip(upper=20.0)
    for col in [c for c in df.columns if "deviation" in c]:
        df[col] = df[col].fillna(0.0)

    # Catch-all: fill any remaining numeric feature NaN with median
    for col in FEATURES:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Drop rows still missing any feature value
    feature_cols = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    df = df.sort_values(["group_id", "time_idx"])

    return df[df["arrival_time"] > "2025-12-23"].copy()


def build_instances(test_df):
    """Build 10 test windows per group. Each window is 20 observations."""
    instances = []
    for group_id in GROUP_IDS:
        gdf = test_df[test_df["group_id"] == group_id].sort_values("time_idx")
        if len(gdf) < WINDOW_SIZE + 1:
            logger.error("%s: only %d rows, need %d", group_id, len(gdf), WINDOW_SIZE + 1)
            sys.exit(1)

        step = max(1, (len(gdf) - WINDOW_SIZE) // SAMPLES_PER_GROUP)
        count = 0
        for offset in range(0, len(gdf) - WINDOW_SIZE, step):
            if count >= SAMPLES_PER_GROUP:
                break
            window = gdf.iloc[offset : offset + WINDOW_SIZE]
            actual = float(window.iloc[-1]["service_headway"])

            observations = []
            for _, row in window.iterrows():
                obs = {}
                for f in FEATURES:
                    if f in row.index:
                        v = row[f]
                        obs[f] = int(v) if isinstance(v, np.integer) else (
                            float(v) if isinstance(v, np.floating) else v)
                observations.append(obs)

            instances.append(({"group_id": group_id, "observations": observations}, actual))
            count += 1

    logger.info("Built %d test instances (%d per group)", len(instances), SAMPLES_PER_GROUP)
    return instances


def run_tests(endpoint, instances):
    """Send instances to endpoint, report MAE / sMAPE / latency."""
    rows = []
    nan_instances = []

    for i, (instance, actual) in enumerate(instances):
        t0 = time.time()
        pred = endpoint.predict(instances=[instance]).predictions[0]
        latency = (time.time() - t0) * 1000

        p10 = pred.get("headway_p10")
        p50 = pred.get("headway_p50")
        p90 = pred.get("headway_p90")

        # Check for NaN or None in the response
        values_bad = any(
            v is None or (isinstance(v, float) and np.isnan(v))
            for v in [p10, p50, p90]
        )

        if values_bad:
            nan_instances.append({
                "index": i,
                "group": instance["group_id"],
                "actual": actual,
                "p10": p10, "p50": p50, "p90": p90,
                "error": pred.get("error"),
                "latency_ms": latency,
            })
        else:
            rows.append({
                "group": instance["group_id"],
                "actual": actual,
                "p10": p10, "p50": p50, "p90": p90,
                "latency_ms": latency,
            })

        if (i + 1) % 10 == 0:
            logger.info("  %d / %d", i + 1, len(instances))

    # --- Report ---
    df = pd.DataFrame(rows)
    df["err"] = (df["p50"] - df["actual"]).abs()
    df["smape"] = 200 * df["err"] / (df["p50"].abs() + df["actual"].abs() + 1e-8)

    mae = df["err"].mean()
    smape = df["smape"].mean()
    p95_lat = df["latency_ms"].quantile(0.95)

    print("\n" + "=" * 60)
    print(" ENDPOINT ACCURACY REPORT")
    print("=" * 60)
    print(f"  Instances sent:   {len(instances)}")
    print(f"  Predictions OK:   {len(rows)}")
    print(f"  Predictions NaN:  {len(nan_instances)}")
    print(f"  MAE:              {mae:.4f} min")
    print(f"  sMAPE:            {smape:.2f}%")
    print(f"  Latency p95:      {p95_lat:.0f} ms")

    print("\n  Per-group breakdown:")
    print(f"    {'Group':<16} {'MAE':>8} {'sMAPE':>8} {'n':>4}")
    for g, gdf in df.groupby("group"):
        print(f"    {g:<16} {gdf['err'].mean():8.4f} {gdf['smape'].mean():7.2f}% {len(gdf):4d}")

    print(f"\n  Sample predictions:")
    print(f"    {'Group':<16} {'Actual':>7} {'P10':>7} {'P50':>7} {'P90':>7} {'Error':>7}")
    for _, r in df.head(10).iterrows():
        print(f"    {r['group']:<16} {r['actual']:7.2f} {r['p10']:7.2f} "
              f"{r['p50']:7.2f} {r['p90']:7.2f} {r['err']:7.2f}")

    # NaN instances
    if nan_instances:
        print(f"\n  NaN predictions ({len(nan_instances)}):")
        print(f"    {'#':>3} {'Group':<16} {'Actual':>7} {'P10':>7} {'P50':>7} {'P90':>7}  Error")
        for rec in nan_instances:
            def _fmt(v):
                if v is None:
                    return "   None"
                if isinstance(v, float) and np.isnan(v):
                    return "    NaN"
                return f"{v:7.2f}"
            print(f"    {rec['index']:3d} {rec['group']:<16} {rec['actual']:7.2f} "
                  f"{_fmt(rec['p10'])} {_fmt(rec['p50'])} {_fmt(rec['p90'])}  "
                  f"{rec.get('error') or ''}")

    passed = mae <= 1.5 and smape <= 25
    print(f"\n  {'PASS' if passed else 'FAIL'}  (thresholds: MAE <= 1.5, sMAPE <= 25%)")
    print("=" * 60)
    return passed


def main():
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"')
    if not endpoints:
        logger.error("No endpoint found: %s", ENDPOINT_DISPLAY_NAME)
        sys.exit(1)
    endpoint = endpoints[0]
    logger.info("Endpoint: %s", endpoint.resource_name)

    test_df = load_test_data()
    logger.info("Test data: %d rows", len(test_df))

    instances = build_instances(test_df)
    passed = run_tests(endpoint, instances)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
