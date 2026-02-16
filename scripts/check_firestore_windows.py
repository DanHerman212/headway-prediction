#!/usr/bin/env python3
"""
check_firestore_windows.py — Monitor the headway_windows Firestore collection.

Shows the current state of each group_id window: observation count,
latest arrival time, latest headway value, and last-updated timestamp.

Usage:
    # One-shot snapshot
    python scripts/check_firestore_windows.py

    # Poll every 30 seconds
    python scripts/check_firestore_windows.py --watch --interval 30
"""

import argparse
import time
from datetime import datetime

from google.cloud import firestore


COLLECTION = "headway_windows"
DATABASE = "headway-streaming"


def fetch_windows(project: str) -> list[dict]:
    """Read all documents from the headway_windows collection."""
    db = firestore.Client(project=project, database=DATABASE)
    docs = db.collection(COLLECTION).stream()
    results = []
    for doc in docs:
        data = doc.to_dict()
        data["_doc_id"] = doc.id
        results.append(data)
    return results


def format_window(doc: dict) -> str:
    """Format a single window document for display."""
    group_id = doc.get("group_id", doc.get("_doc_id", "?"))
    observations = doc.get("observations", [])
    obs_count = len(observations)
    updated_at = doc.get("updated_at")

    if updated_at:
        if hasattr(updated_at, "isoformat"):
            updated_str = updated_at.isoformat()
        else:
            updated_str = str(updated_at)
    else:
        updated_str = "N/A"

    # Extract latest observation details
    if observations:
        latest = observations[-1]
        arrival_time = latest.get("arrival_time", "?")
        if isinstance(arrival_time, (int, float)):
            arrival_time = datetime.fromtimestamp(arrival_time).strftime(
                "%H:%M:%S"
            )
        service_hw = latest.get("service_headway")
        hw_str = f"{service_hw:.1f} min" if service_hw else "N/A"
        route = latest.get("route_id", "?")
        track = latest.get("track_id", "?")
    else:
        arrival_time = "N/A"
        hw_str = "N/A"
        route = "?"
        track = "?"

    return (
        f"  {group_id:12s} | obs: {obs_count:2d}/20 | "
        f"last_arrival: {arrival_time} | headway: {hw_str:>10s} | "
        f"route: {route} | track: {track} | updated: {updated_str}"
    )


def print_snapshot(project: str):
    """Print a formatted snapshot of all window documents."""
    windows = fetch_windows(project)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'=' * 90}")
    print(f"  Firestore Collection: {COLLECTION}    @  {now}")
    print(f"  Documents: {len(windows)}")
    print(f"{'=' * 90}")

    if not windows:
        print("  (no documents found)")
    else:
        windows.sort(key=lambda d: d.get("group_id", ""))
        for doc in windows:
            print(format_window(doc))

    print(f"{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor headway_windows Firestore collection"
    )
    parser.add_argument(
        "--project",
        default="realtime-headway-prediction",
        help="GCP project ID",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll and display updates",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Polling interval in seconds (used with --watch)",
    )
    args = parser.parse_args()

    if args.watch:
        print(f"Watching {COLLECTION} (every {args.interval}s). Ctrl-C to stop.\n")
        try:
            while True:
                print_snapshot(args.project)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_snapshot(args.project)


if __name__ == "__main__":
    main()
