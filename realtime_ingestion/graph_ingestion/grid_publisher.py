"""
Production grid publisher for Graph WaveNet.

Runs on a GCE VM.  Polls the MTA ACE feed every ~30s, keeps the latest
state in memory, and at the top of each minute publishes exactly 137
JSON rows (the dense grid) as a single Pub/Sub message.

Uses the same polling + track-cache + grid-build logic as
local_grid_experiment.py but outputs to Pub/Sub instead of CSV.

Usage:
    python -m realtime_ingestion.graph_ingestion.grid_publisher \
        --project realtime-headway-prediction \
        --topic graph-wavenet-snapshots
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone, timedelta

# On the VM the tarball is flat: grid_publisher.py, gtfs_realtime_pb2.py, etc.
# sit in the same directory.  Locally they live in ../poller/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_POLLER_DIR = os.path.join(_HERE, "..", "poller")
if os.path.isdir(_POLLER_DIR):
    sys.path.insert(0, _POLLER_DIR)
else:
    sys.path.insert(0, _HERE)

import requests
import gtfs_realtime_pb2
import nyct_subway_pb2
from google.cloud import pubsub_v1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

FEED_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"
ACE_ROUTES = {"A", "C", "E"}
STATUS_MAP = {0: "INCOMING_AT", 1: "STOPPED_AT", 2: "IN_TRANSIT_TO"}

# -- Shared mutable state --------------------------------------------------
_lock = threading.Lock()
_vehicles = {}
_track_cache = {}
_feed_ts = None

shutdown_requested = False


def _handle_signal(signum, frame):
    global shutdown_requested
    log.info("Shutdown signal received")
    shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def poll_once():
    """Fetch ACE feed, update shared state. Returns True on success."""
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

    return True


def build_grid(golden_nodes, snapshot_time):
    """Snapshot state, return list of 137 row dicts."""
    with _lock:
        vehicles = dict(_vehicles)
        track_cache = dict(_track_cache)

    occupied = {}
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
            "route_id": info["route_id"] if info else None,
            "trip_id": info["trip_id"] if info else None,
        })
    return rows


def poller_loop(interval):
    """Background thread: poll every `interval` seconds."""
    while not shutdown_requested:
        poll_once()
        for _ in range(int(interval)):
            if shutdown_requested:
                break
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Graph WaveNet grid publisher")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--topic", default="graph-wavenet-snapshots",
                        help="Pub/Sub topic name (default: graph-wavenet-snapshots)")
    parser.add_argument("--poll-interval", type=float, default=30)
    # On VM: node_to_id.json is in the same directory as this script.
    # Locally: it's in local_artifacts/.
    _default_dict = os.path.join(_HERE, "node_to_id.json")
    if not os.path.exists(_default_dict):
        _default_dict = os.path.join(_HERE, "..", "..", "local_artifacts", "node_to_id.json")
    parser.add_argument("--node-dict", default=_default_dict)
    args = parser.parse_args()

    topic_path = f"projects/{args.project}/topics/{args.topic}"
    publisher = pubsub_v1.PublisherClient()

    with open(args.node_dict) as f:
        golden_nodes = json.load(f)
    log.info("Loaded %d golden nodes", len(golden_nodes))

    # Seed state
    log.info("Seeding state...")
    for _ in range(5):
        if poll_once():
            break
        time.sleep(3)
    else:
        log.error("Could not seed state -- exiting")
        sys.exit(1)

    with _lock:
        log.info("State seeded: %d vehicles, %d track entries",
                 len(_vehicles), len(_track_cache))

    # Start background poller
    t = threading.Thread(target=poller_loop, args=(args.poll_interval,), daemon=True)
    t.start()

    log.info("Publishing to %s", topic_path)
    log.info("Waiting for next minute boundary...")

    tick = 0
    while not shutdown_requested:
        now = datetime.now(timezone.utc)
        next_min = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        wait = (next_min - now).total_seconds()
        for _ in range(int(wait)):
            if shutdown_requested:
                break
            time.sleep(1)
        if shutdown_requested:
            break

        snap_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        rows = build_grid(golden_nodes, snap_time)

        payload = json.dumps(rows).encode("utf-8")
        future = publisher.publish(topic_path, payload)
        future.result(timeout=30)

        tick += 1
        occupied = sum(1 for r in rows if r["train_present"] == 1)
        log.info("Tick %d | %s UTC | %d/%d occupied | %d bytes published",
                 tick, snap_time.strftime("%H:%M"), occupied, len(rows), len(payload))

    log.info("Stopped after %d ticks", tick)


if __name__ == "__main__":
    main()
