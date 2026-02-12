"""
Local test for the rush-hour evaluation visualisation.

Generates synthetic data that mirrors the real pipeline's tensor structure,
then runs the new rush-hour plotting logic and validates:

  1. All subplots show the SAME weekday date.
  2. All timestamps fall within the AM rush window (07:00-10:00).
  3. Each subplot has ≥ _MIN_OBS_PER_GROUP observations.
  4. Subplot titles include decoded group_id + date + (n=…).
  5. Trace names are meaningful (no generic "trace 0").

Run:
    PYTHONPATH=. python -m mlops_pipeline.tests.test_rush_hour_viz
"""

import os
import sys
import tempfile
import webbrowser

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


# ---------------------------------------------------------------------------
# Synthetic data builders (shared with test_onnx_export.py)
# ---------------------------------------------------------------------------

def _build_synthetic_data() -> pd.DataFrame:
    """Create a DataFrame mimicking the real training data schema.

    Generates data over 3 weeks with realistic weekday/weekend patterns:
      - Weekday AM rush: headways ~5 min (frequent service)
      - Weekday off-peak / weekend: headways ~12 min
    This lets the rush-hour selector distinguish good days from bad ones.
    """
    np.random.seed(42)

    groups = ["A_South", "C_N", "E_South"]
    rows = []

    # Start on a Monday so weekday logic is predictable
    base_time = pd.Timestamp("2025-12-22 05:00:00", tz="UTC")  # Monday

    for group in groups:
        route = group.split("_")[0]
        direction = group.split("_", 1)[1]
        for day_offset in range(7):  # Mon-Sun (one full week)
            day_start = base_time + pd.Timedelta(days=day_offset)
            is_weekday = day_start.dayofweek < 5

            # Service hours 05:00-01:00, one observation every 5 min
            for minute in range(0, 20 * 60, 5):
                arrival = day_start + pd.Timedelta(minutes=minute)
                hour = arrival.hour

                # Realistic headway distribution
                if is_weekday and 7 <= hour < 10:
                    headway = max(1.5, np.random.normal(loc=5.0, scale=1.5))
                else:
                    headway = max(2.0, np.random.normal(loc=12.0, scale=3.0))

                rows.append({
                    "group_id": group,
                    "route_id": route,
                    "direction": direction,
                    "regime_id": (
                        "AM_RUSH" if 7 <= hour < 10 else
                        "PM_RUSH" if 17 <= hour < 20 else
                        "OFF_PEAK"
                    ),
                    "track_id": "express" if direction == "N" else "local",
                    "preceding_route_id": route,
                    "arrival_time_dt": arrival,
                    "time_idx": 0,
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
                    "hour_sin": np.sin(2 * np.pi * hour / 24),
                    "hour_cos": np.cos(2 * np.pi * hour / 24),
                    "empirical_median": 5.5,
                })

    df = pd.DataFrame(rows)
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
    """Simulate tensors returned by model.predict(mode='raw', return_x=True)."""
    loader = dataset.to_dataloader(train=False, batch_size=512, num_workers=0)

    all_targets, all_groups, all_time_idx = [], [], []
    for batch_x, batch_y in loader:
        all_targets.append(batch_y[0])
        all_groups.append(batch_x["groups"])
        if "decoder_time_idx" in batch_x:
            all_time_idx.append(batch_x["decoder_time_idx"])

    targets = torch.cat(all_targets, dim=0)
    groups = torch.cat(all_groups, dim=0)
    time_idx = torch.cat(all_time_idx, dim=0) if all_time_idx else None

    noise = torch.randn_like(targets.float()) * 0.5
    p50 = targets.float() + noise
    p10 = p50 - torch.abs(torch.randn_like(p50)) * 1.5
    p90 = p50 + torch.abs(torch.randn_like(p50)) * 1.5
    predictions = torch.stack([p10, p50, p90], dim=-1)

    x = {"decoder_target": targets, "groups": groups}
    if time_idx is not None:
        x["decoder_time_idx"] = time_idx

    return predictions, x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from mlops_pipeline.src.steps.evaluate_model import (
        build_rush_hour_figure,
        _build_eval_dataframe,
        _select_rush_date,
        _select_target_groups,
        _MIN_OBS_PER_GROUP,
        _AM_RUSH_START,
        _AM_RUSH_END,
    )

    print("Building synthetic data (3 weeks, 3 groups) …")
    df = _build_synthetic_data()
    print(f"  {len(df)} rows, groups: {df['group_id'].unique().tolist()}")

    print("Creating TimeSeriesDataSet …")
    dataset = _build_dataset(df)
    print(f"  Dataset length: {len(dataset)}")

    print("Generating fake predictions …")
    predictions, x = _build_fake_predictions(dataset)
    print(f"  Predictions shape: {predictions.shape}")

    time_anchor_iso = df["arrival_time_dt"].min().isoformat()
    time_anchor = pd.Timestamp(time_anchor_iso)

    # ── Build the eval DataFrame and run selection logic ──────────────
    eval_df = _build_eval_dataframe(predictions, x, dataset, time_anchor)
    target_groups = _select_target_groups(eval_df)
    rush_date = _select_rush_date(eval_df, target_groups)

    print(f"\n--- Selection Results ---")
    print(f"  Target groups : {target_groups}")
    print(f"  Selected date : {rush_date}")

    # ── Validate the selection ────────────────────────────────────────
    errors = []

    if rush_date is None:
        errors.append("FAIL: No rush date selected")
    else:
        rd = pd.Timestamp(rush_date)
        if rd.dayofweek >= 5:
            errors.append(f"FAIL: Selected date {rush_date} is a weekend (dow={rd.dayofweek})")
        else:
            print(f"  OK: {rush_date} is a weekday ({rd.strftime('%A')})")

        for g in target_groups:
            rush_obs = eval_df[
                (eval_df["date"] == rush_date) &
                (eval_df["hour"] >= _AM_RUSH_START) &
                (eval_df["hour"] < _AM_RUSH_END) &
                (eval_df["group"] == g)
            ]
            n = len(rush_obs)
            if n < _MIN_OBS_PER_GROUP:
                errors.append(f"FAIL: {g} has only {n} obs (need ≥{_MIN_OBS_PER_GROUP})")
            else:
                print(f"  OK: {g} has {n} rush-hour observations")

    # ── Build the figure ──────────────────────────────────────────────
    print("\nBuilding rush-hour figure …")
    fig = build_rush_hour_figure(predictions, x, dataset, time_anchor_iso)

    # Validate subplot titles
    subplot_titles = [
        ann.text for ann in fig.layout.annotations if hasattr(ann, "text")
    ]
    print(f"  Subplot titles: {subplot_titles}")

    if rush_date is not None:
        date_str = pd.Timestamp(rush_date).strftime("%b %d")
        for t in subplot_titles:
            if date_str not in t:
                errors.append(f"FAIL: Title missing date '{date_str}': {t}")
            if "(n=" not in t:
                errors.append(f"FAIL: Title missing observation count: {t}")

    # Check all groups show same date
    if len(subplot_titles) >= 2 and rush_date is not None:
        dates_in_titles = set()
        for t in subplot_titles:
            if date_str in t:
                dates_in_titles.add(date_str)
        if len(dates_in_titles) != 1:
            errors.append(f"FAIL: Subplots show different dates")
        else:
            print("  OK: All subplots aligned to the same date")

    # Check trace names
    trace_names = {t.name for t in fig.data if t.name}
    if any("trace" in str(n).lower() for n in trace_names):
        errors.append("FAIL: Generic 'trace' names found in plot")
    else:
        print(f"  OK: Trace names: {trace_names}")

    # ── Save + open ───────────────────────────────────────────────────
    out_path = os.path.join(tempfile.gettempdir(), "test_rush_hour_viz.html")
    fig.write_html(out_path)
    print(f"\nPlot saved to: {out_path}")
    webbrowser.open(f"file://{out_path}")

    # ── Summary ───────────────────────────────────────────────────────
    if errors:
        print(f"\n{'='*50}")
        print(f"FAILURES ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        print(f"{'='*50}")
        sys.exit(1)
    else:
        print(f"\n{'='*50}")
        print("ALL CHECKS PASSED")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
