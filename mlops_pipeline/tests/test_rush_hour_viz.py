"""
Local test for RushHourVisualizer.

Generates synthetic data that mirrors the real pipeline's tensor structure,
then produces the interactive HTML plot for visual inspection.

Run:
    python -m mlops_pipeline.tests.test_rush_hour_viz

Opens the plot in your browser. Verify:
  1. Subplot titles show decoded group_id strings (e.g. "A_south_local")
  2. All subplots have hover labels: "Predicted (P50)", "90% Confidence", "Actual Headway"
  3. X-axis shows real timestamps (e.g. "Dec 20 08:30")
"""

import os
import sys
import tempfile
import webbrowser

import numpy as np
import pandas as pd
import torch

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


def _build_synthetic_data() -> pd.DataFrame:
    """Create a DataFrame that mimics the real training data schema."""
    np.random.seed(42)

    groups = ["A_south_local", "C_north_express", "E_south_local"]
    rows = []

    # Generate 3 hours of data per group, ~1 observation per minute
    base_time = pd.Timestamp("2025-12-20 07:00:00", tz="UTC")
    for group in groups:
        route = group.split("_")[0]
        n_points = 200  # ~200 minutes of data
        for t in range(n_points):
            arrival = base_time + pd.Timedelta(minutes=t)
            headway = np.random.normal(loc=6.0, scale=2.0)
            headway = max(1.0, headway)
            rows.append({
                "group_id": group,
                "route_id": route,
                "direction": group.split("_")[1],
                "regime_id": "AM_RUSH" if 7 <= arrival.hour < 10 else "MIDDAY",
                "track_id": "express" if "express" in group else "local",
                "preceding_route_id": route,
                "arrival_time_dt": arrival,
                "time_idx": t,  # Will be recomputed below
                "service_headway": headway,
                "preceding_train_gap": headway + np.random.normal(0, 0.5),
                "upstream_headway_14th": headway + np.random.normal(0, 0.3),
                "travel_time_14th": np.random.normal(2.0, 0.3),
                "travel_time_14th_deviation": np.random.normal(0, 0.1),
                "travel_time_23rd": np.random.normal(1.8, 0.2),
                "travel_time_23rd_deviation": np.random.normal(0, 0.1),
                "travel_time_34th": np.random.normal(2.2, 0.3),
                "travel_time_34th_deviation": np.random.normal(0, 0.1),
                "stops_at_23rd": 1.0,
                "hour_sin": np.sin(2 * np.pi * arrival.hour / 24),
                "hour_cos": np.cos(2 * np.pi * arrival.hour / 24),
                "empirical_median": 5.5,
            })

    df = pd.DataFrame(rows)

    # Compute time_idx as minutes from global min (matches data_processing.py)
    min_time = df["arrival_time_dt"].min()
    df["time_idx"] = ((df["arrival_time_dt"] - min_time).dt.total_seconds() / 60).astype(int)
    df = df.sort_values(["group_id", "time_idx"])

    return df


def _build_dataset(df: pd.DataFrame) -> TimeSeriesDataSet:
    """Create a TimeSeriesDataSet matching the real pipeline's schema."""
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="service_headway",
        group_ids=["group_id"],
        min_encoder_length=10,
        max_encoder_length=20,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["route_id"],
        time_varying_known_categoricals=["regime_id", "track_id"],
        time_varying_known_reals=["time_idx", "hour_sin", "hour_cos", "empirical_median"],
        time_varying_unknown_categoricals=["preceding_route_id"],
        time_varying_unknown_reals=[
            "service_headway",
            "preceding_train_gap",
            "upstream_headway_14th",
            "travel_time_14th",
            "travel_time_14th_deviation",
            "travel_time_23rd",
            "travel_time_23rd_deviation",
            "travel_time_34th",
            "travel_time_34th_deviation",
            "stops_at_23rd",
        ],
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def _build_fake_predictions(dataset: TimeSeriesDataSet):
    """Simulate the tensor structure returned by model.predict(mode='raw', return_x=True)."""
    loader = dataset.to_dataloader(train=False, batch_size=512, num_workers=0)

    all_targets = []
    all_groups = []
    all_time_idx = []

    for batch_x, batch_y in loader:
        all_targets.append(batch_y[0])  # decoder_target
        all_groups.append(batch_x["groups"])
        if "decoder_time_idx" in batch_x:
            all_time_idx.append(batch_x["decoder_time_idx"])

    targets = torch.cat(all_targets, dim=0)          # (N, pred_len)
    groups = torch.cat(all_groups, dim=0)             # (N, 1)
    time_idx = torch.cat(all_time_idx, dim=0) if all_time_idx else None

    # Fake quantile predictions: P50 ≈ actual + noise, P10 below, P90 above
    noise = torch.randn_like(targets.float()) * 0.5
    p50 = targets.float() + noise
    p10 = p50 - torch.abs(torch.randn_like(p50)) * 1.5
    p90 = p50 + torch.abs(torch.randn_like(p50)) * 1.5

    predictions = torch.stack([p10, p50, p90], dim=-1)  # (N, pred_len, 3)

    x = {
        "decoder_target": targets,
        "groups": groups,
    }
    if time_idx is not None:
        x["decoder_time_idx"] = time_idx

    return predictions, x


def main():
    print("Building synthetic data...")
    df = _build_synthetic_data()
    print(f"  {len(df)} rows, groups: {df['group_id'].unique().tolist()}")

    print("Creating TimeSeriesDataSet...")
    dataset = _build_dataset(df)
    print(f"  Dataset length: {len(dataset)}")

    print("Generating fake predictions...")
    predictions, x = _build_fake_predictions(dataset)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Groups in x: {x['groups'].unique().tolist()}")

    if "decoder_time_idx" in x:
        t = x["decoder_time_idx"].view(-1)
        print(f"  time_idx range: [{t.min().item()}, {t.max().item()}]")

    print("Creating RushHourVisualizer...")
    from mlops_pipeline.src.steps.evaluate_model import RushHourVisualizer

    time_anchor_iso = df["arrival_time_dt"].min().isoformat()
    print(f"  time_anchor_iso: {time_anchor_iso}")

    viz = RushHourVisualizer(predictions, x, dataset, time_anchor_iso)
    fig = viz.plot_rush_hour(window_size=180)

    # Save and open
    out_path = os.path.join(tempfile.gettempdir(), "test_rush_hour_viz.html")
    fig.write_html(out_path)
    print(f"\nPlot saved to: {out_path}")
    print("Opening in browser...")
    webbrowser.open(f"file://{out_path}")

    # Print checks
    print("\n--- Verification Checklist ---")
    subplot_titles = [ann.text for ann in fig.layout.annotations if hasattr(ann, "text")]
    print(f"  Subplot titles: {subplot_titles}")
    has_numeric_title = any(t.isdigit() for t in subplot_titles)
    if has_numeric_title:
        print("  FAIL: Subplot titles still show numeric IDs")
    else:
        print("  OK: Subplot titles show decoded group names")

    trace_names = [t.name for t in fig.data if t.name]
    print(f"  Trace names: {set(trace_names)}")
    if any("trace" in str(n).lower() for n in trace_names):
        print("  FAIL: Generic 'trace' names found")
    else:
        print("  OK: All traces have meaningful names")

    # Check x-axis data for timestamps
    first_data_trace = next((t for t in fig.data if t.x is not None and len(t.x) > 0), None)
    if first_data_trace is not None:
        sample = first_data_trace.x[0]
        if isinstance(sample, (pd.Timestamp, np.datetime64)) or "T" in str(sample):
            print(f"  OK: X-axis uses timestamps (sample: {sample})")
        else:
            print(f"  FAIL: X-axis still uses raw values (sample: {sample})")
    else:
        print("  WARN: No trace data found to check x-axis")
    print()


if __name__ == "__main__":
    main()
