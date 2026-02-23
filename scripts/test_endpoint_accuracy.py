#!/usr/bin/env python3
"""
test_endpoint_accuracy.py
-------------------------
Comprehensive integration test for the deployed headway-prediction endpoint.

Sends real historical observations from the TEST split (unseen during training)
to the live Vertex AI endpoint and compares predictions against known actuals.

Tests performed:
  1. Smoke test      – single payload returns valid response structure
  2. All groups      – each group_id produces reasonable predictions
  3. Accuracy test   – MAE/sMAPE across many samples vs training-eval baseline
  4. Quantile order  – P10 ≤ P50 ≤ P90 for every prediction
  5. Range test      – predictions fall within plausible headway bounds
  6. Latency test    – response time is within acceptable limits

Usage:
  python scripts/test_endpoint_accuracy.py                  # full test
  python scripts/test_endpoint_accuracy.py --samples 10     # quick smoke
  python scripts/test_endpoint_accuracy.py --dry-run        # local model only
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
PROJECT_ID = "realtime-headway-prediction"
LOCATION = "us-east1"
ENDPOINT_DISPLAY_NAME = "headway-prediction-endpoint"

# Thresholds (based on training eval: MAE ≈ 0.34, sMAPE ≈ 4.2%)
MAE_THRESHOLD = 1.5          # fail if endpoint MAE exceeds this (generous margin)
SMAPE_THRESHOLD = 25.0       # fail if sMAPE exceeds this %
LATENCY_P95_MS = 5000        # per-request P95 latency limit (ms)
HEADWAY_MIN = 0.0            # minimum plausible headway (minutes)
HEADWAY_MAX = 60.0           # maximum plausible headway (minutes)

ENCODER_LENGTH = 19           # send 20 observations; last becomes decoder target
GROUP_IDS = ["A_South", "C_South", "E_South", "OTHER_South"]

OBSERVATION_FEATURES = [
    "service_headway", "preceding_train_gap", "upstream_headway_14th",
    "travel_time_14th", "travel_time_14th_deviation",
    "travel_time_23rd", "travel_time_23rd_deviation",
    "travel_time_34th", "travel_time_34th_deviation",
    "stops_at_23rd", "hour_sin", "hour_cos", "day_of_week",
    "time_idx", "empirical_median",
    "route_id", "regime_id", "track_id", "preceding_route_id",
]


# ──────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────

def load_test_data() -> pd.DataFrame:
    """Load and clean the test split from training data."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mlops_pipeline.src.data_processing import clean_dataset

    path = os.path.join(
        os.path.dirname(__file__), "..", "local_artifacts",
        "processed_data", "training_data.parquet",
    )
    raw_df = pd.read_parquet(path)
    clean_df, _ = clean_dataset(raw_df)

    # Use only the test split (after val_end_date = 2025-12-23)
    if "arrival_time" in clean_df.columns:
        clean_df["arrival_time"] = pd.to_datetime(clean_df["arrival_time"])
        test_df = clean_df[clean_df["arrival_time"] > "2025-12-23"].copy()
    else:
        # Fall back: use the last 20% by time_idx
        cutoff = clean_df["time_idx"].quantile(0.8)
        test_df = clean_df[clean_df["time_idx"] > cutoff].copy()

    logger.info(
        "Test data: %d rows across %d groups",
        len(test_df), test_df["group_id"].nunique(),
    )
    return test_df


def build_instances(
    test_df: pd.DataFrame,
    samples_per_group: int = 10,
) -> List[Tuple[Dict[str, Any], float]]:
    """Build (payload_instance, actual_headway) pairs from test data.

    For each sample we take 20 consecutive observations.  The LAST
    observation's service_headway is the ground-truth target — the
    endpoint should predict something close to it.
    """
    instances = []
    skipped_groups = []
    for group_id in GROUP_IDS:
        gdf = test_df[test_df["group_id"] == group_id].sort_values("time_idx")
        if len(gdf) < 21:
            logger.warning("Skipping %s — only %d rows", group_id, len(gdf))
            skipped_groups.append(group_id)
            continue

        # Evenly space sample windows across the group's test data
        n = min(samples_per_group, max(1, (len(gdf) - 20) // 20))
        step = max(1, (len(gdf) - 20) // n)
        offsets = list(range(0, min(n * step, len(gdf) - 20), step))

        for offset in offsets:
            window = gdf.iloc[offset : offset + 20]
            actual = float(window.iloc[-1]["service_headway"])

            observations = []
            for _, row in window.iterrows():
                obs = {}
                for feat in OBSERVATION_FEATURES:
                    if feat in row.index:
                        val = row[feat]
                        # Convert numpy types to native Python for JSON
                        if isinstance(val, (np.integer,)):
                            val = int(val)
                        elif isinstance(val, (np.floating,)):
                            val = float(val)
                        obs[feat] = val
                observations.append(obs)

            instance = {"group_id": group_id, "observations": observations}
            instances.append((instance, actual))

    logger.info("Built %d test instances (skipped groups: %s)", len(instances), skipped_groups or "none")
    return instances, skipped_groups


# ──────────────────────────────────────────────────────────────
# Endpoint caller
# ──────────────────────────────────────────────────────────────

def get_endpoint():
    """Find the deployed endpoint."""
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"',
    )
    if not endpoints:
        logger.error("No endpoint found with display_name='%s'", ENDPOINT_DISPLAY_NAME)
        sys.exit(1)
    endpoint = endpoints[0]
    logger.info("Using endpoint: %s", endpoint.resource_name)
    return endpoint


def predict_one(endpoint, instance: Dict[str, Any]) -> Tuple[Dict, float]:
    """Send a single instance to the endpoint. Returns (result_dict, latency_ms)."""
    t0 = time.time()
    response = endpoint.predict(instances=[instance])
    latency_ms = (time.time() - t0) * 1000

    preds = response.predictions
    if not preds or len(preds) == 0:
        return {"error": "empty response"}, latency_ms

    return preds[0], latency_ms


# ──────────────────────────────────────────────────────────────
# Local model fallback (--dry-run)
# ──────────────────────────────────────────────────────────────

def predict_local(instance: Dict[str, Any]) -> Tuple[Dict, float]:
    """Run prediction locally using the deployed model artifacts."""
    import torch
    from pytorch_forecasting import TimeSeriesDataSet

    ds = torch.load("/tmp/deployed_training_dataset.pt", weights_only=False, map_location="cpu")
    model = torch.load("/tmp/deployed_model_full.pt", weights_only=False, map_location="cpu")
    model.eval()

    group_id = instance["group_id"]
    df = pd.DataFrame(instance["observations"])
    df["group_id"] = group_id
    for col in ["group_id", "route_id", "regime_id", "track_id", "preceding_route_id"]:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)
    df["time_idx"] = df["time_idx"].astype(int)

    t0 = time.time()
    predict_ds = TimeSeriesDataSet.from_dataset(ds, df, predict=True, stop_randomization=True)
    dl = predict_ds.to_dataloader(batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        preds = model.predict(dl, mode="quantiles")

    latency_ms = (time.time() - t0) * 1000
    q = preds[0, 0, :].cpu().numpy()

    return {
        "group_id": group_id,
        "headway_p10": round(float(q[0]), 2),
        "headway_p50": round(float(q[1]), 2),
        "headway_p90": round(float(q[2]), 2),
    }, latency_ms


# ──────────────────────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────────────────────

def run_tests(
    instances: List[Tuple[Dict, float]],
    predict_fn,
    max_samples: Optional[int] = None,
    skipped_groups: Optional[List[str]] = None,
) -> bool:
    """Run all tests. Returns True if all pass."""
    if max_samples:
        instances = instances[:max_samples]

    results = []  # (group_id, predicted_p50, actual, p10, p50, p90, latency_ms)
    errors = []

    for i, (instance, actual) in enumerate(instances):
        try:
            pred, latency = predict_fn(instance)
        except Exception as e:
            errors.append(f"Instance {i} ({instance['group_id']}): {e}")
            continue

        if "error" in pred:
            errors.append(f"Instance {i} ({instance['group_id']}): {pred['error']}")
            continue

        results.append({
            "group_id": instance["group_id"],
            "actual": actual,
            "p10": pred["headway_p10"],
            "p50": pred["headway_p50"],
            "p90": pred["headway_p90"],
            "latency_ms": latency,
        })

        if (i + 1) % 5 == 0 or i == len(instances) - 1:
            logger.info(
                "  [%d/%d] %s: pred=%.2f actual=%.2f err=%.2f (%.0fms)",
                i + 1, len(instances),
                instance["group_id"],
                pred["headway_p50"], actual,
                abs(pred["headway_p50"] - actual),
                latency,
            )

    if not results:
        logger.error("No successful predictions at all!")
        return False

    df = pd.DataFrame(results)
    all_passed = True

    # ── Test 1: Smoke — got at least some results ──
    print("\n" + "=" * 65)
    print(" TEST RESULTS")
    print("=" * 65)

    n_ok = len(results)
    n_err = len(errors)
    # Allow up to 30% errors — some windows land on sparse time_idx regions
    # that pytorch-forecasting's encoder filter rejects on the server side
    smoke_pass = n_ok > 0 and n_err <= 0.3 * (n_ok + n_err)
    _report("1. Smoke test", smoke_pass,
            f"{n_ok} succeeded, {n_err} errors")
    if not smoke_pass:
        all_passed = False
    for e in errors[:5]:
        print(f"     ERROR: {e}")

    # ── Test 2: All groups represented (excluding groups with insufficient test data) ──
    groups_seen = set(df["group_id"].unique())
    expected_groups = set(GROUP_IDS) - set(skipped_groups or [])
    groups_pass = groups_seen == expected_groups
    detail = f"seen={sorted(groups_seen)}, expected={sorted(expected_groups)}"
    if skipped_groups:
        detail += f" (skipped {skipped_groups}: insufficient test data)"
    _report("2. All groups", groups_pass, detail)
    if not groups_pass:
        all_passed = False

    # ── Test 3: Accuracy — MAE & sMAPE ──
    df["abs_err"] = (df["p50"] - df["actual"]).abs()
    df["smape"] = 200 * df["abs_err"] / (df["p50"].abs() + df["actual"].abs() + 1e-8)
    mae = df["abs_err"].mean()
    smape = df["smape"].mean()
    mae_pass = mae <= MAE_THRESHOLD
    smape_pass = smape <= SMAPE_THRESHOLD
    _report("3a. MAE", mae_pass,
            f"{mae:.4f} min (threshold {MAE_THRESHOLD})")
    _report("3b. sMAPE", smape_pass,
            f"{smape:.2f}% (threshold {SMAPE_THRESHOLD}%)")
    if not mae_pass or not smape_pass:
        all_passed = False

    # Per-group breakdown
    print("\n     Per-group MAE:")
    for gid, gdf in df.groupby("group_id"):
        g_mae = gdf["abs_err"].mean()
        g_n = len(gdf)
        print(f"       {gid:15s}  MAE={g_mae:.4f}  (n={g_n})")

    # ── Test 4: Quantile ordering — P10 ≤ P50 ≤ P90 ──
    ordering_ok = ((df["p10"] <= df["p50"] + 0.01) & (df["p50"] <= df["p90"] + 0.01))
    n_violated = (~ordering_ok).sum()
    quant_pass = n_violated == 0
    _report("4. Quantile order", quant_pass,
            f"{n_violated}/{len(df)} violations (P10 ≤ P50 ≤ P90)")
    if not quant_pass:
        all_passed = False
        violators = df[~ordering_ok].head(3)
        for _, row in violators.iterrows():
            print(f"     VIOLATION: {row['group_id']} "
                  f"P10={row['p10']:.2f} P50={row['p50']:.2f} P90={row['p90']:.2f}")

    # ── Test 5: Plausible range ──
    in_range = (df["p50"] >= HEADWAY_MIN) & (df["p50"] <= HEADWAY_MAX)
    n_oor = (~in_range).sum()
    range_pass = n_oor == 0
    _report("5. Plausible range", range_pass,
            f"{n_oor}/{len(df)} out of [{HEADWAY_MIN}, {HEADWAY_MAX}] min")
    if not range_pass:
        all_passed = False

    # ── Test 6: Latency ──
    p50_lat = df["latency_ms"].median()
    p95_lat = df["latency_ms"].quantile(0.95)
    lat_pass = p95_lat <= LATENCY_P95_MS
    _report("6. Latency", lat_pass,
            f"p50={p50_lat:.0f}ms  p95={p95_lat:.0f}ms (limit {LATENCY_P95_MS}ms)")
    if not lat_pass:
        all_passed = False

    # ── Summary ──
    print("\n" + "-" * 65)
    print(f" Overall MAE:   {mae:.4f} min")
    print(f" Overall sMAPE: {smape:.2f}%")
    print(f" Predictions:   {n_ok} succeeded / {n_err} errors")
    print(f" Median latency: {p50_lat:.0f} ms")

    if all_passed:
        print("\n ✅  ALL TESTS PASSED")
    else:
        print("\n ❌  SOME TESTS FAILED")
    print("=" * 65)

    # ── Detailed predictions table ──
    print("\nSample predictions (first 15):")
    print(f"  {'Group':<16} {'Actual':>7} {'P10':>7} {'P50':>7} {'P90':>7} {'Error':>7}")
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for _, row in df.head(15).iterrows():
        print(f"  {row['group_id']:<16} {row['actual']:7.2f} "
              f"{row['p10']:7.2f} {row['p50']:7.2f} {row['p90']:7.2f} "
              f"{row['abs_err']:7.2f}")

    return all_passed


def _report(name: str, passed: bool, detail: str):
    icon = "✅" if passed else "❌"
    print(f"  {icon}  {name}: {detail}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comprehensive endpoint accuracy test")
    parser.add_argument("--samples", type=int, default=None,
                        help="Max total samples (default: ~10 per group)")
    parser.add_argument("--samples-per-group", type=int, default=10,
                        help="Samples per group (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test locally instead of hitting the endpoint")
    args = parser.parse_args()

    logger.info("Loading test data...")
    test_df = load_test_data()

    logger.info("Building test instances...")
    instances, skipped_groups = build_instances(test_df, samples_per_group=args.samples_per_group)

    if args.dry_run:
        logger.info("DRY RUN — using local model artifacts")
        predict_fn = lambda inst: predict_local(inst)
    else:
        logger.info("Connecting to Vertex AI endpoint...")
        endpoint = get_endpoint()
        predict_fn = lambda inst: predict_one(endpoint, inst)

    logger.info("Running tests (%d instances)...\n", len(instances))
    passed = run_tests(instances, predict_fn, max_samples=args.samples, skipped_groups=skipped_groups)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
