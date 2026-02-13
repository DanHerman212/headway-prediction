"""
test_endpoint.py
----------------
Send a test prediction request to the deployed headway prediction endpoint.

Usage:
    python scripts/test_endpoint.py [--endpoint-id ENDPOINT_ID]

If --endpoint-id is not provided, looks up the endpoint by display name.
"""

import argparse
import json
import math
import logging

from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT = "realtime-headway-prediction"
LOCATION = "us-east1"
ENDPOINT_DISPLAY_NAME = "headway-prediction-endpoint"

# ---- Realistic test payloads for each group_id ----

def _build_test_payload(group_id: str, route: str, direction: str) -> dict:
    """Build a 20-observation encoder window for a rush hour scenario."""
    observations = []
    base_time_idx = 14400  # ~10 days into dataset, arbitrary
    for i in range(20):
        t = base_time_idx + i
        hour = 8 + (i / 60)  # ~8:00-8:20 AM rush
        headway = 5.0 + (i % 3) * 0.5  # 5.0-6.0 min range
        observations.append({
            "time_idx": t,
            "service_headway": round(headway, 2),
            "preceding_train_gap": round(headway + 0.3, 2),
            "upstream_headway_14th": round(headway - 0.2, 2),
            "travel_time_14th": 2.1,
            "travel_time_14th_deviation": 0.05,
            "travel_time_23rd": 1.8,
            "travel_time_23rd_deviation": 0.03,
            "travel_time_34th": 2.2,
            "travel_time_34th_deviation": 0.04,
            "stops_at_23rd": 1.0,
            "hour_sin": round(math.sin(2 * math.pi * hour / 24), 4),
            "hour_cos": round(math.cos(2 * math.pi * hour / 24), 4),
            "empirical_median": 5.5,
            "route_id": route,
            "regime_id": "Night" if (hour >= 22 or hour < 5) else "Day",
            "track_id": "local" if direction == "South" else "express",
            "preceding_route_id": route,
        })

    return {
        "group_id": group_id,
        "observations": observations,
    }


TEST_INSTANCES = [
    _build_test_payload("A_South", "A", "South"),
    _build_test_payload("C_N", "C", "N"),
    _build_test_payload("E_South", "E", "South"),
]


def main():
    parser = argparse.ArgumentParser(description="Test headway prediction endpoint")
    parser.add_argument("--endpoint-id", type=str, default=None,
                        help="Vertex AI Endpoint resource ID")
    args = parser.parse_args()

    aiplatform.init(project=PROJECT, location=LOCATION)

    # Find endpoint
    if args.endpoint_id:
        endpoint = aiplatform.Endpoint(args.endpoint_id)
    else:
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"'
        )
        if not endpoints:
            logger.error("No endpoint found with display name '%s'", ENDPOINT_DISPLAY_NAME)
            return
        endpoint = endpoints[0]

    logger.info("Using endpoint: %s", endpoint.resource_name)

    # Send predictions for all 3 groups
    request_body = {"instances": TEST_INSTANCES}

    logger.info("Sending %d test instances...", len(TEST_INSTANCES))
    response = endpoint.predict(instances=TEST_INSTANCES)

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    all_pass = True
    for pred in response.predictions:
        group = pred["group_id"]
        p10 = pred["headway_p10"]
        p50 = pred["headway_p50"]
        p90 = pred["headway_p90"]

        # Validation checks
        checks = []
        if p10 < p50 < p90:
            checks.append("OK: P10 < P50 < P90")
        else:
            checks.append("FAIL: quantile ordering broken")
            all_pass = False

        if 0 < p50 < 30:
            checks.append("OK: P50 in sane range (0-30 min)")
        else:
            checks.append(f"FAIL: P50={p50} outside expected range")
            all_pass = False

        print(f"\n  {group}:")
        print(f"    P10={p10:.2f}  P50={p50:.2f}  P90={p90:.2f}")
        for c in checks:
            print(f"    {c}")

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
