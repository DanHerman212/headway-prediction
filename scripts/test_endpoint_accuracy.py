#!/usr/bin/env python3
"""
test_endpoint_accuracy.py
-------------------------
Post-deployment accuracy test for the headway prediction endpoint.

Loads the training parquet, extracts encoder windows from the test
period, sends them to the live Vertex AI endpoint, and compares the
P50 predictions against ground-truth service_headway values.

Reports:
  - Overall MAE and sMAPE
  - Per-group MAE breakdown
  - Quantile calibration (% of actuals below P10, P50, P90)
  - Pass / fail verdict against configurable MAE threshold

Usage:
  python scripts/test_endpoint_accuracy.py
  python scripts/test_endpoint_accuracy.py --samples 200
  python scripts/test_endpoint_accuracy.py --mae-threshold 2.0
  python scripts/test_endpoint_accuracy.py --local-data local_artifacts/raw_data/training_data.parquet
"""

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from google.cloud import aiplatform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants (match deploy_endpoint.py) ──────────────────────
PROJECT_ID = "realtime-headway-prediction"
LOCATION = "us-east1"
ENDPOINT_DISPLAY_NAME = "headway-prediction-endpoint"
TRAINING_DATA_URI = (
    "gs://mlops-artifacts-realtime-headway-prediction/data/training_data.parquet"
)

# Encoder window expected by the model
ENCODER_LENGTH = 20

# Test split boundary (observations on or after this date are test data)
TEST_START_DATE = "2025-12-23"

# Columns the endpoint expects per observation
OBSERVATION_COLS = [
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
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "time_idx",
    "empirical_median",
    "route_id",
    "regime_id",
    "track_id",
    "preceding_route_id",
]


# ── Data preparation ─────────────────────────────────────────


def load_and_clean(data_path: str) -> pd.DataFrame:
    """Load the training parquet and apply the same cleaning as the pipeline.

    Mirrors mlops_pipeline.src.data_processing.clean_dataset() so that
    observation values sent to the endpoint match what the model was
    trained on (no NaNs, clipped outliers, etc.).
    """
    logger.info("Loading data from %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Parse dates
    if "arrival_time" in df.columns:
        df["arrival_time_dt"] = pd.to_datetime(df["arrival_time"])

    # Categoricals to string (matches clean_dataset)
    for col in ["group_id", "route_id", "direction", "regime_id", "track_id", "preceding_route_id"]:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    # Impute missing numerics with column median (matches clean_dataset)
    for col in ["preceding_train_gap", "upstream_headway_14th", "travel_time_14th", "travel_time_34th"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Clip preceding_train_gap outliers at 20 min (matches clean_dataset)
    if "preceding_train_gap" in df.columns:
        df["preceding_train_gap"] = df["preceding_train_gap"].clip(upper=20.0)

    # Fill deviation columns with 0.0 — assume on-time if unknown
    for col in [c for c in df.columns if "deviation" in c]:
        df[col] = df[col].fillna(0.0)

    # Fill travel_time_23rd with median (matches clean_dataset)
    if "travel_time_23rd" in df.columns:
        df["travel_time_23rd"] = df["travel_time_23rd"].fillna(df["travel_time_23rd"].median())

    # Compute time_idx if missing
    if "time_idx" not in df.columns:
        min_time = df["arrival_time_dt"].min()
        df["time_idx"] = (
            (df["arrival_time_dt"] - min_time).dt.total_seconds() / 60
        ).astype(int)

    df = df.sort_values(["group_id", "time_idx"])
    return df


def extract_test_windows(
    df: pd.DataFrame,
    n_samples: int,
    seed: int = 42,
) -> List[Tuple[Dict[str, Any], float]]:
    """Extract (instance, ground_truth) pairs from the test period.

    For each sample we take ENCODER_LENGTH consecutive observations as the
    encoder window and the very next observation's service_headway as ground
    truth.

    Returns a list of (instance_dict, actual_headway) tuples.
    """
    is_tz_aware = df["arrival_time_dt"].dt.tz is not None
    tz = "UTC" if is_tz_aware else None
    test_start = pd.Timestamp(TEST_START_DATE, tz=tz)

    # Collect all valid windows per group
    candidates: List[Tuple[str, int]] = []  # (group_id, iloc_start_of_target)
    for group_id, gdf in df.groupby("group_id"):
        # Find rows in the test period
        test_mask = gdf["arrival_time_dt"] >= test_start
        test_indices = gdf.index[test_mask]
        if len(test_indices) == 0:
            continue

        # For each test-period row, check that there are ENCODER_LENGTH
        # preceding rows in the same group to form the encoder window.
        iloc_positions = gdf.index.get_indexer(test_indices)
        for pos in iloc_positions:
            if pos >= ENCODER_LENGTH:
                candidates.append((group_id, gdf.index[pos]))

    if not candidates:
        logger.error("No valid test windows found. Check TEST_START_DATE and data.")
        sys.exit(1)

    logger.info("Found %d candidate test windows across all groups", len(candidates))

    # Sample
    rng = np.random.RandomState(seed)
    n = min(n_samples, len(candidates))
    chosen = rng.choice(len(candidates), size=n, replace=False)

    windows = []
    for idx in chosen:
        group_id, target_idx = candidates[idx]
        gdf = df.loc[df["group_id"] == group_id].copy()
        iloc_in_group = gdf.index.get_loc(target_idx)

        # Encoder window: ENCODER_LENGTH rows ending just before the target
        window_df = gdf.iloc[iloc_in_group - ENCODER_LENGTH : iloc_in_group]
        actual = float(gdf.iloc[iloc_in_group]["service_headway"])

        # Build the instance dict matching the endpoint's expected format
        observations = []
        for _, row in window_df.iterrows():
            obs = {}
            for col in OBSERVATION_COLS:
                val = row[col]
                # Convert numpy types to native Python for JSON serialization
                if hasattr(val, "item"):
                    val = val.item()
                obs[col] = val
            observations.append(obs)

        instance = {
            "group_id": group_id,
            "observations": observations,
        }
        windows.append((instance, actual))

    logger.info("Sampled %d test windows", len(windows))
    return windows


# ── Endpoint interaction ──────────────────────────────────────


def get_endpoint() -> aiplatform.Endpoint:
    """Resolve the deployed prediction endpoint."""
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"',
    )
    if not endpoints:
        logger.error(
            "No endpoint found with display_name='%s'. "
            "Deploy with: python scripts/deploy_endpoint.py",
            ENDPOINT_DISPLAY_NAME,
        )
        sys.exit(1)
    endpoint = endpoints[0]
    logger.info("Using endpoint: %s", endpoint.resource_name)
    return endpoint


def predict_batch(
    endpoint: aiplatform.Endpoint,
    instances: List[Dict[str, Any]],
    batch_size: int = 5,
) -> List[Dict[str, Any]]:
    """Send instances to the endpoint in batches, return predictions."""
    all_predictions = []
    n_batches = (len(instances) + batch_size - 1) // batch_size

    for i in range(0, len(instances), batch_size):
        batch = instances[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info("Sending batch %d/%d (%d instances)", batch_num, n_batches, len(batch))

        start = time.time()
        response = endpoint.predict(instances=batch)
        elapsed = (time.time() - start) * 1000
        logger.info("  Response in %.0f ms", elapsed)

        all_predictions.extend(response.predictions)

    return all_predictions


# ── Metrics ───────────────────────────────────────────────────


def compute_metrics(
    actuals: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    groups: List[str],
) -> Dict[str, Any]:
    """Compute accuracy and calibration metrics."""
    errors = np.abs(actuals - p50)
    mae = float(np.mean(errors))

    # sMAPE: symmetric mean absolute percentage error
    denom = (np.abs(actuals) + np.abs(p50)) / 2.0
    denom = np.where(denom == 0, 1e-8, denom)
    smape = float(np.mean(np.abs(actuals - p50) / denom) * 100)

    # Quantile calibration
    pct_below_p10 = float(np.mean(actuals < p10) * 100)
    pct_below_p50 = float(np.mean(actuals < p50) * 100)
    pct_below_p90 = float(np.mean(actuals < p90) * 100)

    # Per-group MAE
    group_arr = np.array(groups)
    per_group_mae = {}
    for g in sorted(set(groups)):
        mask = group_arr == g
        per_group_mae[g] = float(np.mean(errors[mask]))

    return {
        "mae": mae,
        "smape": smape,
        "n_samples": len(actuals),
        "quantile_calibration": {
            "pct_below_p10": round(pct_below_p10, 1),
            "pct_below_p50": round(pct_below_p50, 1),
            "pct_below_p90": round(pct_below_p90, 1),
            "ideal_p10": 10.0,
            "ideal_p50": 50.0,
            "ideal_p90": 90.0,
        },
        "per_group_mae": per_group_mae,
    }


# ── Main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Test deployed headway prediction endpoint against ground truth"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of test windows to sample (default: 100)",
    )
    parser.add_argument(
        "--mae-threshold",
        type=float,
        default=2.5,
        help="Maximum acceptable MAE in minutes (default: 2.5). "
        "Test fails if endpoint MAE exceeds this.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of instances per endpoint request (default: 5)",
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Path to a local parquet file instead of fetching from GCS",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    args = parser.parse_args()

    # ── Init ──────────────────────────────────────────────────
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # ── Load data ─────────────────────────────────────────────
    data_path = args.local_data or TRAINING_DATA_URI
    df = load_and_clean(data_path)

    # ── Build test windows ────────────────────────────────────
    windows = extract_test_windows(df, n_samples=args.samples, seed=args.seed)
    instances = [w[0] for w in windows]
    actuals = np.array([w[1] for w in windows])
    groups = [w[0]["group_id"] for w in windows]

    # ── Send to endpoint ──────────────────────────────────────
    endpoint = get_endpoint()
    logger.info("=" * 60)
    logger.info("Sending %d test instances to endpoint", len(instances))
    logger.info("=" * 60)

    predictions = predict_batch(endpoint, instances, batch_size=args.batch_size)

    # ── Parse results ─────────────────────────────────────────
    p10_list, p50_list, p90_list = [], [], []
    errors_list = []
    for i, pred in enumerate(predictions):
        if pred.get("error"):
            logger.warning(
                "Instance %d (%s) returned error: %s",
                i, groups[i], pred["error"],
            )
            # Use NaN so this sample is excluded from metrics
            p10_list.append(np.nan)
            p50_list.append(np.nan)
            p90_list.append(np.nan)
            continue
        p10_list.append(pred["headway_p10"])
        p50_list.append(pred["headway_p50"])
        p90_list.append(pred["headway_p90"])

    p10 = np.array(p10_list)
    p50 = np.array(p50_list)
    p90 = np.array(p90_list)

    # Drop instances with errors
    valid = ~np.isnan(p50)
    n_errors = int((~valid).sum())
    if n_errors:
        logger.warning("%d / %d instances returned errors — excluded from metrics", n_errors, len(p50))
    actuals = actuals[valid]
    p10 = p10[valid]
    p50 = p50[valid]
    p90 = p90[valid]
    groups_valid = [g for g, v in zip(groups, valid) if v]

    if len(actuals) == 0:
        logger.error("All instances failed. Cannot compute metrics.")
        sys.exit(1)

    # ── Compute metrics ───────────────────────────────────────
    metrics = compute_metrics(actuals, p10, p50, p90, groups_valid)

    # ── Report ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ENDPOINT ACCURACY TEST RESULTS")
    logger.info("=" * 60)
    logger.info("  Samples:  %d  (errors: %d)", metrics["n_samples"], n_errors)
    logger.info("  MAE:      %.4f minutes", metrics["mae"])
    logger.info("  sMAPE:    %.2f%%", metrics["smape"])
    logger.info("")
    logger.info("  Quantile calibration (ideal → actual):")
    cal = metrics["quantile_calibration"]
    logger.info("    P10: %.0f%% ideal → %.1f%% actual", cal["ideal_p10"], cal["pct_below_p10"])
    logger.info("    P50: %.0f%% ideal → %.1f%% actual", cal["ideal_p50"], cal["pct_below_p50"])
    logger.info("    P90: %.0f%% ideal → %.1f%% actual", cal["ideal_p90"], cal["pct_below_p90"])
    logger.info("")
    logger.info("  Per-group MAE:")
    for g, g_mae in sorted(metrics["per_group_mae"].items()):
        logger.info("    %-12s  %.4f min", g, g_mae)

    # ── Pass / Fail ───────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    if metrics["mae"] <= args.mae_threshold:
        logger.info(
            "PASS — MAE %.4f <= threshold %.2f minutes",
            metrics["mae"],
            args.mae_threshold,
        )
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error(
            "FAIL — MAE %.4f > threshold %.2f minutes",
            metrics["mae"],
            args.mae_threshold,
        )
        logger.info("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
