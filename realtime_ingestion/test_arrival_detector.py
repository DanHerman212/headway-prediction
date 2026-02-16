"""
Test the ArrivalDetector against the saved snapshot and optionally live feed.

Usage:
    # Test against saved snapshot only
    python3 realtime_ingestion/test_arrival_detector.py

    # Test against live feed (2 polls, 30s apart)
    python3 realtime_ingestion/test_arrival_detector.py --live --polls 2
"""

import argparse
import json
import logging
import os
import sys
import time

# Add project root to path for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipelines.beam.transforms.arrival_detector import ArrivalDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "ace_snapshot.json")


def test_snapshot():
    """Test against the saved snapshot file.

    Since we only have one snapshot, we simulate two polls with the same data.
    First poll: seeds state + treats all STOPPED_AT as arrivals.
    Second poll: same data = no new arrivals (validates diffing logic).
    """
    print("=" * 60)
    print("TEST 1: Single snapshot — seed arrivals")
    print("=" * 60)

    with open(SNAPSHOT_PATH) as f:
        feed = json.load(f)

    detector = ArrivalDetector()

    # --- Poll 1: first poll, all STOPPED_AT become arrivals ---
    arrivals = detector.process_feed(feed)
    print(f"\nPoll 1: {len(arrivals)} arrivals detected (first poll = all STOPPED_AT)")

    # Breakdown
    by_route = {}
    with_track = 0
    without_track = 0
    for a in arrivals:
        by_route[a["route_id"]] = by_route.get(a["route_id"], 0) + 1
        if a["track"]:
            with_track += 1
        else:
            without_track += 1

    print(f"  By route: {by_route}")
    print(f"  With track: {with_track},  Without track: {without_track}")

    # Show corridor arrivals
    corridor = {"A28S", "A29S", "A30S", "A31S", "A32S"}
    corridor_arrivals = [a for a in arrivals if a["stop_id"] in corridor]
    print(f"\n  Corridor arrivals (A28S-A32S): {len(corridor_arrivals)}")
    for a in corridor_arrivals:
        print(f"    {a}")

    # Show a few sample records in baseline format
    print(f"\n  Sample baseline records (first 5):")
    for a in arrivals[:5]:
        print(f"    {json.dumps(a, indent=6)}")

    # --- Poll 2: same data, should produce ZERO new arrivals ---
    print(f"\n{'=' * 60}")
    print("TEST 2: Repeat same snapshot — should detect 0 new arrivals")
    print("=" * 60)

    arrivals2 = detector.process_feed(feed)
    print(f"\nPoll 2: {len(arrivals2)} arrivals detected")
    assert len(arrivals2) == 0, f"Expected 0 new arrivals, got {len(arrivals2)}"
    print("  ✓ Correctly detected no new arrivals on duplicate poll")

    return True


def test_live(num_polls: int = 2, poll_interval: int = 30):
    """Test against live MTA feed with multiple polls.

    This validates the real arrival detection: poll 1 seeds, poll 2+ detect
    actual new arrivals as trains move.
    """
    # Import protobuf parsing from fetch_snapshot
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "poller"))
    import requests
    import gtfs_realtime_pb2
    import nyct_subway_pb2

    URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"

    def fetch_and_parse() -> dict:
        """Fetch live feed and parse to the same JSON dict format as snapshot."""
        from datetime import datetime, timezone as tz

        resp = requests.get(URL, timeout=10)
        resp.raise_for_status()
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(resp.content)

        feed_dict = {
            "header": {
                "gtfs_realtime_version": feed.header.gtfs_realtime_version,
                "timestamp": feed.header.timestamp,
            },
            "entity": [],
        }

        for entity in feed.entity:
            ed = {"id": entity.id}

            if entity.HasField("trip_update"):
                tu = entity.trip_update
                td = {
                    "trip_id": tu.trip.trip_id,
                    "route_id": tu.trip.route_id if tu.trip.HasField("route_id") else None,
                    "start_time": tu.trip.start_time if tu.trip.HasField("start_time") else None,
                    "start_date": tu.trip.start_date if tu.trip.HasField("start_date") else None,
                }
                if tu.trip.HasExtension(nyct_subway_pb2.nyct_trip_descriptor):
                    nyct = tu.trip.Extensions[nyct_subway_pb2.nyct_trip_descriptor]
                    td["train_id"] = nyct.train_id if nyct.HasField("train_id") else None
                    td["direction"] = (
                        nyct.Direction.Name(nyct.direction)
                        if nyct.HasField("direction") else None
                    )

                stops = []
                for su in tu.stop_time_update:
                    sd = {"stop_id": su.stop_id}
                    if su.HasField("arrival"):
                        sd["arrival_time"] = su.arrival.time
                    if su.HasField("departure"):
                        sd["departure_time"] = su.departure.time
                    if su.HasExtension(nyct_subway_pb2.nyct_stop_time_update):
                        ns = su.Extensions[nyct_subway_pb2.nyct_stop_time_update]
                        sd["scheduled_track"] = ns.scheduled_track if ns.HasField("scheduled_track") else None
                        sd["actual_track"] = ns.actual_track if ns.HasField("actual_track") else None
                    stops.append(sd)

                ed["trip_update"] = {"trip": td, "stop_time_update": stops}

            elif entity.HasField("vehicle"):
                v = entity.vehicle
                ed["vehicle"] = {
                    "trip_id": v.trip.trip_id if v.trip.HasField("trip_id") else None,
                    "route_id": v.trip.route_id if v.trip.HasField("route_id") else None,
                    "stop_id": v.stop_id if v.HasField("stop_id") else None,
                    "current_status": (
                        v.VehicleStopStatus.Name(v.current_status)
                        if v.HasField("current_status") else None
                    ),
                    "timestamp": v.timestamp if v.HasField("timestamp") else None,
                }

            feed_dict["entity"].append(ed)

        return feed_dict

    print("=" * 60)
    print(f"LIVE TEST: {num_polls} polls, {poll_interval}s interval")
    print("=" * 60)

    detector = ArrivalDetector()

    for i in range(num_polls):
        print(f"\n--- Poll {i + 1}/{num_polls} ---")
        try:
            feed = fetch_and_parse()
            entities = feed.get("entity", [])
            tu_count = sum(1 for e in entities if "trip_update" in e)
            vp_count = sum(1 for e in entities if "vehicle" in e)
            print(f"  Feed timestamp: {feed['header']['timestamp']}")
            print(f"  Entities: {tu_count} trip_updates, {vp_count} vehicles")

            arrivals = detector.process_feed(feed)
            print(f"  New arrivals: {len(arrivals)}")

            if arrivals:
                # Show corridor arrivals
                corridor = {"A28S", "A29S", "A30S", "A31S", "A32S"}
                corridor_arr = [a for a in arrivals if a["stop_id"] in corridor]
                if corridor_arr:
                    print(f"  Corridor arrivals: {len(corridor_arr)}")
                    for a in corridor_arr:
                        print(f"    {a}")

                # Track resolution rate for this poll
                with_track = sum(1 for a in arrivals if a["track"])
                print(f"  Track resolved: {with_track}/{len(arrivals)}")

        except Exception as e:
            print(f"  ERROR: {e}")

        if i < num_polls - 1:
            print(f"\n  Waiting {poll_interval}s for next poll...")
            time.sleep(poll_interval)

    print(f"\n{'=' * 60}")
    print("LIVE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ArrivalDetector")
    parser.add_argument("--live", action="store_true", help="Test against live MTA feed")
    parser.add_argument("--polls", type=int, default=3, help="Number of live polls (default 3)")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between polls (default 30)")
    args = parser.parse_args()

    # Always run snapshot test first
    ok = test_snapshot()

    if args.live:
        print("\n\n")
        test_live(num_polls=args.polls, poll_interval=args.interval)
