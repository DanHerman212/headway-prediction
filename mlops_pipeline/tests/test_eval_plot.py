"""
test_eval_plot.py
-----------------
Local test for the lean evaluation step prediction plot.

Generates synthetic data covering the target window
(Jan 14, 2026 15:00-21:00 ET) and validates that the plot
contains the expected routes and observation counts.
"""

import sys
import os
import tempfile

import numpy as np
import pandas as pd
import torch

# Constants matching evaluate_model.py
PLOT_DATE = "2026-01-14"
PLOT_HOUR_START = 15
PLOT_HOUR_END = 21
PLOT_TZ = "America/New_York"

# Known distribution from user's dataset
EXPECTED_ROUTES = {"A": 62, "C": 33, "E": 71}


def _build_synthetic_data():
    """Build synthetic data covering the target plot window."""
    # Time anchor: fixed point before the test set
    time_anchor = pd.Timestamp("2025-10-13 15:00:00", tz="UTC")

    # Target window in UTC (ET + 5h in January = EST)
    # 15:00 ET = 20:00 UTC, 21:00 ET = 02:00 UTC next day
    window_start_utc = pd.Timestamp(f"{PLOT_DATE} 20:00:00", tz="UTC")
    window_end_utc = pd.Timestamp("2026-01-15 02:00:00", tz="UTC")

    groups = {
        "A_South": EXPECTED_ROUTES["A"],
        "C_South": EXPECTED_ROUTES["C"],
        "E_South": EXPECTED_ROUTES["E"],
    }

    rows = []
    for group, count in groups.items():
        times = pd.date_range(window_start_utc, window_end_utc, periods=count + 2)[1:-1]
        for t in times[:count]:
            time_idx = int((t - time_anchor).total_seconds() / 60)
            rows.append({
                "group_id": group,
                "route_id": group.split("_")[0],
                "time_idx": time_idx,
                "service_headway": np.random.uniform(4, 12),
            })

    return pd.DataFrame(rows), time_anchor.isoformat()


def _build_dataset(df):
    """Create a minimal TimeSeriesDataSet from synthetic data."""
    from pytorch_forecasting import TimeSeriesDataSet

    # Add prefix rows for encoder history
    prefix_rows = []
    for gid in df["group_id"].unique():
        gdf = df[df["group_id"] == gid].sort_values("time_idx")
        min_idx = gdf["time_idx"].min()
        for i in range(25):
            prefix_rows.append({
                "group_id": gid,
                "route_id": gid.split("_")[0],
                "time_idx": min_idx - 25 + i,
                "service_headway": np.random.uniform(5, 10),
            })

    full_df = pd.concat([pd.DataFrame(prefix_rows), df]).sort_values(
        ["group_id", "time_idx"]
    ).reset_index(drop=True)

    return TimeSeriesDataSet(
        full_df,
        time_idx="time_idx",
        target="service_headway",
        group_ids=["group_id"],
        max_encoder_length=20,
        max_prediction_length=1,
        min_encoder_length=10,
        static_categoricals=["route_id"],
        time_varying_unknown_reals=["service_headway"],
        allow_missing_timesteps=True,
    )


def _build_fake_predictions(dataset):
    """Generate fake predictions + x tensors matching model output shape."""
    loader = dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    all_targets, all_groups, all_time_idx = [], [], []
    for batch_x, batch_y in loader:
        all_targets.append(batch_y[0])
        all_groups.append(batch_x["groups"])
        all_time_idx.append(batch_x["decoder_time_idx"])

    targets = torch.cat(all_targets, dim=0)
    groups = torch.cat(all_groups, dim=0)
    time_idx = torch.cat(all_time_idx, dim=0)

    B = targets.shape[0]
    noise = torch.randn(B, 1) * 0.5
    p50 = targets + noise
    p10 = p50 - 1.0
    p90 = p50 + 1.0
    predictions = torch.stack([p10, p50, p90], dim=2)

    x = {
        "decoder_target": targets,
        "groups": groups,
        "decoder_time_idx": time_idx,
    }
    return predictions, x


def main():
    print("Building synthetic data ...")
    df, time_anchor_iso = _build_synthetic_data()
    print(f"  Rows: {len(df)}")

    print("Building dataset ...")
    dataset = _build_dataset(df)

    print("Building fake predictions ...")
    predictions, x = _build_fake_predictions(dataset)
    print(f"  Prediction shape: {predictions.shape}")

    # Import functions under test
    from mlops_pipeline.src.steps.evaluate_model import (
        _build_eval_dataframe,
        build_prediction_figure,
    )

    time_anchor = pd.Timestamp(time_anchor_iso)
    eval_df = _build_eval_dataframe(predictions, x, dataset, time_anchor)
    print(f"Eval DataFrame: {len(eval_df)} rows")
    print(f"  Routes: {sorted(eval_df['route'].unique())}")

    # Apply same window filter as production code
    eval_df_et = eval_df.copy()
    ts = eval_df_et["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    eval_df_et["ts_et"] = ts.dt.tz_convert(PLOT_TZ)
    start = pd.Timestamp(f"{PLOT_DATE} {PLOT_HOUR_START:02d}:00", tz=PLOT_TZ)
    end = pd.Timestamp(f"{PLOT_DATE} {PLOT_HOUR_END:02d}:00", tz=PLOT_TZ)
    window = eval_df_et[(eval_df_et["ts_et"] >= start) & (eval_df_et["ts_et"] < end)]

    route_counts = window.groupby("route").size()
    print(f"\nWindow ({PLOT_DATE} {PLOT_HOUR_START}:00-{PLOT_HOUR_END}:00 ET):")
    print(f"  Total rows: {len(window)}")
    for r, c in route_counts.items():
        print(f"    {r}: {c}")

    # Build the figure
    fig = build_prediction_figure(eval_df)

    # --- Assertions ---
    errors = []

    routes_in_plot = sorted(window["route"].unique())
    if routes_in_plot != ["A", "C", "E"]:
        errors.append(f"Expected routes [A, C, E], got {routes_in_plot}")

    for route, expected in EXPECTED_ROUTES.items():
        actual_count = route_counts.get(route, 0)
        if actual_count < expected * 0.5:
            errors.append(f"Route {route}: expected ~{expected}, got {actual_count}")

    if len(fig.data) == 0:
        errors.append("Figure has no traces")

    titles = [a.text for a in fig.layout.annotations if hasattr(a, "text")]
    for t in titles:
        if PLOT_DATE not in t:
            errors.append(f"Subplot title missing date: {t}")

    tmp = os.path.join(tempfile.gettempdir(), "test_eval_plot.html")
    fig.write_html(tmp)
    print(f"\nPlot saved to: {tmp}")

    if errors:
        print("\n*** FAILURES ***")
        for e in errors:
            print(f"  FAIL: {e}")
        sys.exit(1)
    else:
        print("\n*** ALL CHECKS PASSED ***")


if __name__ == "__main__":
    main()
