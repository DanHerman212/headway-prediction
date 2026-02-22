#!/usr/bin/env python3
"""
test_cpr_local.py
-----------------
Locally test the CPR HeadwayPredictor without Docker or Vertex AI.

Loads the model artifacts from local_artifacts/models/latest/ and runs
the same load → preprocess → predict → postprocess pipeline that the
CPR container executes, using a sample request from test_payload.json.

Usage:
  python scripts/test_cpr_local.py
  python scripts/test_cpr_local.py --payload path/to/custom_payload.json
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "local_artifacts", "models", "latest"
)
DEFAULT_PAYLOAD = os.path.join(
    os.path.dirname(__file__), "..", "local_artifacts", "test_payload.json"
)


def main():
    parser = argparse.ArgumentParser(description="Test CPR predictor locally")
    parser.add_argument(
        "--payload", default=DEFAULT_PAYLOAD,
        help="Path to JSON payload (Vertex AI format with 'instances' key)"
    )
    parser.add_argument(
        "--artifacts-dir", default=ARTIFACTS_DIR,
        help="Path to local model artifacts directory"
    )
    args = parser.parse_args()

    artifacts_dir = os.path.abspath(args.artifacts_dir)
    if not os.path.isdir(artifacts_dir):
        logger.error("Artifacts directory not found: %s", artifacts_dir)
        sys.exit(1)

    # Import and instantiate the predictor
    from mlops_pipeline.src.serving.cpr_predictor import HeadwayPredictor

    predictor = HeadwayPredictor()

    # Step 1: Load
    logger.info("=" * 50)
    logger.info("Step 1: load()")
    logger.info("=" * 50)
    t0 = time.time()
    predictor.load(artifacts_dir)
    logger.info("Load completed in %.1f s", time.time() - t0)

    # Step 2: Load payload
    with open(args.payload) as f:
        payload = json.load(f)

    # Wrap in {"instances": [...]} if needed
    if "instances" not in payload:
        payload = {"instances": [payload]}

    n_instances = len(payload["instances"])
    logger.info("Payload: %d instance(s)", n_instances)

    # Step 3: Preprocess
    logger.info("=" * 50)
    logger.info("Step 2: preprocess()")
    logger.info("=" * 50)
    t0 = time.time()
    prepared = predictor.preprocess(payload)
    logger.info("Preprocess completed in %.3f s — %d prepared instances",
                time.time() - t0, len(prepared))

    # Step 4: Predict
    logger.info("=" * 50)
    logger.info("Step 3: predict()")
    logger.info("=" * 50)
    t0 = time.time()
    results = predictor.predict(prepared)
    elapsed = time.time() - t0
    logger.info("Predict completed in %.3f s", elapsed)

    # Step 5: Postprocess
    logger.info("=" * 50)
    logger.info("Step 4: postprocess()")
    logger.info("=" * 50)
    response = predictor.postprocess(results)

    # Print results
    logger.info("=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    print(json.dumps(response, indent=2))

    # Summary
    for pred in response.get("predictions", []):
        if pred.get("error"):
            logger.warning("  %s: ERROR — %s", pred["group_id"], pred["error"])
        else:
            logger.info(
                "  %s: P10=%.2f  P50=%.2f  P90=%.2f min",
                pred["group_id"],
                pred["headway_p10"],
                pred["headway_p50"],
                pred["headway_p90"],
            )


if __name__ == "__main__":
    main()
