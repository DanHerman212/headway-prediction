"""
Local experiment: Dense 1-minute grid capture for Graph WaveNet.

Polls the MTA ACE feed every ~30s, keeps the latest state in memory,
and at the top of each minute writes exactly 137 CSV rows (one per
golden node) to a local file.

No GCP dependencies. Uses the compiled protobuf modules already in
realtime_ingestion/poller/.

Usage:
    cd realtime_ingestion/graph_ingestion
    python local_grid_experiment.py                        # run for 5 minutes
    python local_grid_experiment.py --ticks 60             # run for 60 minutes
    python local_grid_experiment.py --ticks 0              # run indefinitely
"""

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta

# Add the poller directory so we can import compiled protobuf modules
POLLER_DIR = os.path.join(os.path.dirname(__file__), "..", "poller")
sys.path.insert(0, POLLER_DIR)

import requests
import gtfs_realtime_pb2
import nyct_subway_pb2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

FEED_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"
ACE_ROUTES = {"A", "C", "E"}
STATUS_MAP = {0: "INCOMING_AT", 1: "STOPPED_AT", 2: "IN_TRANSIT_TO"}
CSV_HEADER = ["snapshot_time", "node_id", "train_present", "route_id", "trip_id"]

#  Shared mutable state (written by poller, read by metronome) 
_lock = threading.Lock()
_vehicles = {}          # trip_id -> {stop_id, status, route_id, timestamp}
_track_cache = {}       # (trip_id, stop_id) -> track
_feed_ts = None         # last feed header timestamp


def poll_once():
    """Fetch ACE feed, update _vehicles and _track_cache."""
    global _vehicles, _track_cache, _feed_ts

    try:
        resp = requests.get(FEED_URL, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Fetch failed: %s", e)
        return False

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)

    vehicles = {}
    new_tracks = {}

    for entity in feed.entity:
        #  trip_update  extract track info 
        if entity.HasField("trip_update"):
            tu = entity.trip_update
            route = tu.trip.route_id if tu.trip.HasField("route_id") else None
            if route not in ACE_ROUTES:
                continue
            tid = tu.trip.trip_id
            for su in tu.stop_time_update:
                sid = su.stop_id
                if not sid:
                    continue
                if su.HasExtension(nyct_subway_pb2.nyct_stop_time_update):
                    ns = su.Extensions[nyct_subway_pb2.nyct_stop_time_update]
                    track = None
                    if ns.HasField("actual_track"):
                        track = ns.actual_track
                    elif ns.HasField("scheduled_track"):
                        track = ns.scheduled_track
                    if track:
                        new_tracks[(tid, sid)] = track

        #  vehicle  current position 
        elif entity.HasField("vehicle"):
            v = entity.vehicle
            tid = v.trip.trip_id if v.trip.HasField("trip_id") else None
            route = v.trip.route_id if v.trip.HasField("route_id") else None
            if not tid or route not in ACE_ROUTES:
                continue
            vehicles[tid] = {
                "stop_id": v.stop_id if v.HasField("stop_id") else None,
                "status": STATUS_MAP.get(v.current_status, "UNKNOWN"),
                "route_id": route,
                "timestamp": v.timestamp if v.HasField("timestamp") else None,
            }

    with _lock:
        _vehicles = vehicles
        _track_cache.update(new_tracks)
        _feed_ts = feed.header.timestamp

    log.debug(
        "Poll OK  %d vehicles, %d new track entries, feed_ts=%s",
        len(vehicles), len(new_tracks), _feed_ts,
    )
    return True


def build_grid(golden_nodes, snapshot_time):
    """Snapshot state and return 137 rows for this minute."""
    with _lock:
        vehicles = dict(_vehicles)
        track_cache = dict(_track_cache)

    # Find nodes occupied by a STOPPED_AT train
    occupied = {}  # node_id -> {route_id, trip_id}
    for tid, v in vehicles.items():
        if v["status"] != "STOPPED_AT":
            continue
        sid = v.get("stop_id")
        if not sid:
            continue
        track = track_cache.get((tid, sid))
        if not track:
            continue
        node_id = f"{sid}_{track}"
        if node_id in golden_nodes:
            occupied[node_id] = {"route_id": v["route_id"], "trip_id": tid}

    ts_str = snapshot_time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for node_id in golden_nodes:
        info = occupied.get(node_id)
        rows.append({
            "snapshot_time": ts_str,
            "node_id": node_id,
            "train_present": 1 if info else 0,
            "route_id": info["route_id"] if info else "",
            "trip_id": info["trip_id"] if info else "",
        })
    return rows, len(occupied)


def poller_loop(interval, stop_evt):
    """Background thread: poll every `interval` seconds."""
    while not stop_evt.is_set():
        poll_once()
        stop_evt.wait(timeout=interval)


def main():
    parser = argparse.ArgumentParser(description="Local dense grid experiment")
    parser.add_argument(
        "--ticks", type=int, default=5,
        help="Number of 1-minute ticks to capture (0 = indefinite, default: 5)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=30,
        help="Seconds between feed polls (default: 30)",
    )
    parser.add_argument(
        "--node-dict", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "local_artifacts", "node_to_id.json"
        ),
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "local_artifacts", "dense_grid.csv"
        ),
    )
    args = parser.parse_args()

    # Load golden nodes
    with open(args.node_dict) as f:
        golden_nodes = json.load(f)
    log.info("Loaded %d golden nodes", len(golden_nodes))

    # Seed state with first poll
    log.info("Seeding state...")
    for attempt in range(5):
        if poll_once():
            break
        time.sleep(3)
    else:
        log.error("Could not fetch feed after 5 attempts  exiting")
        sys.exit(1)

    with _lock:
        log.info(
            "State seeded: %d vehicles, %d track-cache entries",
            len(_vehicles), len(_track_cache),
        )

    # Start background poller
    stop_evt = threading.Event()
    t = threading.Thread(target=poller_loop, args=(args.poll_interval, stop_evt), daemon=True)
    t.start()

    # Open CSV for writing
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    file_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    csv_file = open(args.output, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
    if not file_exists:
        writer.writeheader()

    log.info("Writing to %s", args.output)
    log.info("Waiting for next minute boundary...")

    tick = 0
    try:
        while True:
            # Wait until next :00 second boundary
            now = datetime.now(timezone.utc)
            next_min = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
            time.sleep((next_min - now).total_seconds())

            snap_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            rows, n_occupied = build_grid(golden_nodes, snap_time)
            writer.writerows(rows)
            csv_file.flush()

            tick += 1
            log.info(
                "Tick %d | %s UTC | %d/%d nodes occupied",
                tick, snap_time.strftime("%H:%M"), n_occupied, len(golden_nodes),
            )

            if args.ticks > 0 and tick >= args.ticks:
                log.info("Reached %d ticks  done", args.ticks)
                break

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        stop_evt.set()
        csv_file.close()
        log.info("CSV closed. Total ticks: %d, rows: %d", tick, tick * len(golden_nodes))


if __name__ == "__main__":
    main()
