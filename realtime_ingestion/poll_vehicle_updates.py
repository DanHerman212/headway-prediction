"""
Poll MTA GTFS-RT feed every 15 seconds for 1 minute and display vehicle updates.

No GCP credentials required — fetches directly from the public MTA API
and prints vehicle position snapshots to stdout.

Usage:
    python realtime_ingestion/poll_vehicle_updates.py
    python realtime_ingestion/poll_vehicle_updates.py --feed bdfm
    python realtime_ingestion/poll_vehicle_updates.py --interval 10 --duration 30
    python realtime_ingestion/poll_vehicle_updates.py --dump-raw  # save full GTFS snapshots
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "poller"))

import requests
import gtfs_realtime_pb2
import nyct_subway_pb2

# GTFS-RT VehicleStopStatus enum: 0=INCOMING_AT, 1=STOPPED_AT, 2=IN_TRANSIT_TO
STATUS_NAMES = {0: "INCOMING_AT", 1: "STOPPED_AT", 2: "IN_TRANSIT_TO"}

FEEDS = {
    "ace": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace",
    "bdfm": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-bdfm",
}


def _parse_status(vehicle) -> str:
    """Reliably get vehicle status name.

    HasField('current_status') returns False for IN_TRANSIT_TO (enum=2)
    in some protobuf versions, so we always read the raw int and map it.
    """
    return STATUS_NAMES.get(vehicle.current_status, f"UNKNOWN({vehicle.current_status})")


def fetch_and_parse(url: str, save_raw: bool = False) -> dict | None:
    """Fetch a GTFS-RT feed and return parsed dict.

    If save_raw=True, the raw protobuf bytes and full parsed JSON (including
    trip_updates) are included in the result under 'raw_bytes_data' and
    'raw_entities'.
    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  !! Fetch failed: {e}")
        return None

    raw_content = resp.content
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(raw_content)

    result = {
        "header_timestamp": feed.header.timestamp,
        "fetch_time": datetime.now(timezone.utc).isoformat(),
        "raw_bytes": len(raw_content),
        "vehicles": [],
        "trip_update_count": 0,
    }

    if save_raw:
        result["raw_bytes_data"] = raw_content
        result["raw_entities"] = []

    for entity in feed.entity:
        # --- Trip updates ---
        if entity.HasField("trip_update"):
            result["trip_update_count"] += 1

            if save_raw:
                tu = entity.trip_update
                trip = tu.trip
                td = {
                    "trip_id": trip.trip_id,
                    "route_id": trip.route_id if trip.HasField("route_id") else None,
                    "start_time": trip.start_time if trip.HasField("start_time") else None,
                    "start_date": trip.start_date if trip.HasField("start_date") else None,
                }
                if trip.HasExtension(nyct_subway_pb2.nyct_trip_descriptor):
                    nyct = trip.Extensions[nyct_subway_pb2.nyct_trip_descriptor]
                    td["train_id"] = nyct.train_id if nyct.HasField("train_id") else None
                    td["direction"] = nyct.Direction.Name(nyct.direction) if nyct.HasField("direction") else None

                stops = []
                for su in tu.stop_time_update:
                    sd = {"stop_id": su.stop_id}
                    if su.HasField("arrival"):
                        sd["arrival_time"] = su.arrival.time if su.arrival.HasField("time") else None
                    if su.HasField("departure"):
                        sd["departure_time"] = su.departure.time if su.departure.HasField("time") else None
                    if su.HasExtension(nyct_subway_pb2.nyct_stop_time_update):
                        ns = su.Extensions[nyct_subway_pb2.nyct_stop_time_update]
                        sd["scheduled_track"] = ns.scheduled_track if ns.HasField("scheduled_track") else None
                        sd["actual_track"] = ns.actual_track if ns.HasField("actual_track") else None
                    stops.append(sd)

                result["raw_entities"].append({
                    "id": entity.id,
                    "trip_update": {"trip": td, "stop_time_update": stops},
                })

        # --- Vehicle positions ---
        if entity.HasField("vehicle"):
            v = entity.vehicle
            status = _parse_status(v)

            vd = {
                "entity_id": entity.id,
                "trip_id": v.trip.trip_id if v.trip.HasField("trip_id") else None,
                "route_id": v.trip.route_id if v.trip.HasField("route_id") else None,
                "stop_id": v.stop_id if v.HasField("stop_id") else None,
                "current_status": status,
                "timestamp": v.timestamp if v.HasField("timestamp") else None,
                "current_stop_sequence": (
                    v.current_stop_sequence
                    if v.HasField("current_stop_sequence")
                    else None
                ),
            }

            # NYCT trip extension for direction
            if v.trip.HasExtension(nyct_subway_pb2.nyct_trip_descriptor):
                nyct = v.trip.Extensions[nyct_subway_pb2.nyct_trip_descriptor]
                vd["train_id"] = nyct.train_id if nyct.HasField("train_id") else None
                vd["direction"] = (
                    nyct.Direction.Name(nyct.direction)
                    if nyct.HasField("direction")
                    else None
                )

            result["vehicles"].append(vd)

            if save_raw:
                result["raw_entities"].append({"id": entity.id, "vehicle": vd})

    return result


def diff_vehicles(prev_vehicles: list[dict], curr_vehicles: list[dict]) -> dict:
    """Compare two snapshots of vehicle positions and find changes."""
    prev_by_trip = {v["trip_id"]: v for v in prev_vehicles if v["trip_id"]}
    curr_by_trip = {v["trip_id"]: v for v in curr_vehicles if v["trip_id"]}

    prev_ids = set(prev_by_trip.keys())
    curr_ids = set(curr_by_trip.keys())

    new_trips = curr_ids - prev_ids
    removed_trips = prev_ids - curr_ids
    continuing = prev_ids & curr_ids

    moved = []
    status_changed = []
    for tid in continuing:
        p, c = prev_by_trip[tid], curr_by_trip[tid]
        if p["stop_id"] != c["stop_id"]:
            moved.append({
                "trip_id": tid,
                "route_id": c["route_id"],
                "from_stop": p["stop_id"],
                "to_stop": c["stop_id"],
                "prev_status": p["current_status"],
                "curr_status": c["current_status"],
            })
        elif p["current_status"] != c["current_status"]:
            status_changed.append({
                "trip_id": tid,
                "route_id": c["route_id"],
                "stop_id": c["stop_id"],
                "from_status": p["current_status"],
                "to_status": c["current_status"],
            })

    return {
        "new_trips": len(new_trips),
        "removed_trips": len(removed_trips),
        "moved": moved,
        "status_changed": status_changed,
    }


def print_snapshot(poll_num: int, snapshot: dict, prev_vehicles: list[dict] | None):
    """Print a formatted snapshot summary."""
    vehicles = snapshot["vehicles"]
    routes = defaultdict(int)
    statuses = defaultdict(int)
    for v in vehicles:
        routes[v["route_id"]] += 1
        statuses[v["current_status"]] += 1

    print(f"\n{'='*70}")
    print(f"  POLL #{poll_num}  |  {snapshot['fetch_time']}  |  Feed ts: {snapshot['header_timestamp']}")
    print(f"{'='*70}")
    print(f"  Raw bytes: {snapshot['raw_bytes']:,}")
    print(f"  Trip updates: {snapshot['trip_update_count']}  |  Vehicle positions: {len(vehicles)}")
    print(f"  Routes: {dict(sorted(routes.items()))}")
    print(f"  Statuses: {dict(sorted(statuses.items(), key=lambda x: str(x[0])))}")

    if prev_vehicles is not None:
        diff = diff_vehicles(prev_vehicles, vehicles)
        print(f"\n  --- Changes since last poll ---")
        print(f"  New trips: {diff['new_trips']}  |  Removed trips: {diff['removed_trips']}")
        if diff["moved"]:
            print(f"  Vehicles that moved ({len(diff['moved'])}):")
            for m in diff["moved"][:10]:
                print(f"    {m['route_id']} {m['trip_id']}: {m['from_stop']} → {m['to_stop']}  ({m['prev_status']} → {m['curr_status']})")
            if len(diff["moved"]) > 10:
                print(f"    ... and {len(diff['moved']) - 10} more")
        else:
            print(f"  No vehicles moved to a different stop.")
        if diff["status_changed"]:
            print(f"  Status changes ({len(diff['status_changed'])}):")
            for s in diff["status_changed"][:10]:
                print(f"    {s['route_id']} {s['trip_id']} @ {s['stop_id']}: {s['from_status']} → {s['to_status']}")
            if len(diff["status_changed"]) > 10:
                print(f"    ... and {len(diff['status_changed']) - 10} more")
    else:
        # First poll — show a few sample vehicles
        print(f"\n  --- Sample vehicles (first 5) ---")
        for v in vehicles[:5]:
            direction = v.get("direction", "?")
            train = v.get("train_id", "?")
            print(f"    {v['route_id']} | {v['trip_id']} | stop {v['stop_id']} | {v['current_status']} | dir={direction} | train={train}")


def main():
    parser = argparse.ArgumentParser(description="Poll MTA GTFS-RT vehicle updates")
    parser.add_argument("--feed", choices=list(FEEDS.keys()), default="ace", help="Feed to poll (default: ace)")
    parser.add_argument("--interval", type=int, default=15, help="Poll interval in seconds (default: 15)")
    parser.add_argument("--duration", type=int, default=60, help="Total duration in seconds (default: 60)")
    parser.add_argument("--dump-json", action="store_true", help="Dump vehicle-only JSON per poll")
    parser.add_argument("--dump-raw", action="store_true", help="Dump full raw GTFS snapshots (trip_updates + vehicles + protobuf)")
    args = parser.parse_args()

    url = FEEDS[args.feed]
    num_polls = (args.duration // args.interval) + 1
    print(f"Polling {args.feed.upper()} feed every {args.interval}s for {args.duration}s ({num_polls} polls)")
    print(f"URL: {url}")

    prev_vehicles = None
    all_snapshots = []

    save_raw = args.dump_raw
    outdir = os.path.join(os.path.dirname(__file__), "snapshots")

    for i in range(1, num_polls + 1):
        snapshot = fetch_and_parse(url, save_raw=save_raw)
        if snapshot is None:
            print(f"\n  Poll #{i} FAILED — retrying next interval")
        else:
            print_snapshot(i, snapshot, prev_vehicles)
            all_snapshots.append(snapshot)
            prev_vehicles = snapshot["vehicles"]

            if args.dump_json or save_raw:
                os.makedirs(outdir, exist_ok=True)

            if args.dump_json:
                outpath = os.path.join(outdir, f"vehicles_poll{i}.json")
                with open(outpath, "w") as f:
                    json.dump(snapshot["vehicles"], f, indent=2)
                print(f"  Dumped vehicles to {outpath}")

            if save_raw:
                # Save full parsed JSON (trip_updates + vehicles)
                raw_entities = snapshot.pop("raw_entities", [])
                raw_bytes = snapshot.pop("raw_bytes_data", b"")
                raw_snapshot = {
                    "header_timestamp": snapshot["header_timestamp"],
                    "fetch_time": snapshot["fetch_time"],
                    "entity": raw_entities,
                }
                json_path = os.path.join(outdir, f"raw_poll{i}.json")
                with open(json_path, "w") as f:
                    json.dump(raw_snapshot, f, indent=2)

                # Save raw protobuf bytes
                pb_path = os.path.join(outdir, f"raw_poll{i}.pb")
                with open(pb_path, "wb") as f:
                    f.write(raw_bytes)

                print(f"  Dumped raw GTFS to {json_path} ({len(raw_entities)} entities) + {pb_path}")

        if i < num_polls:
            print(f"\n  Waiting {args.interval}s...")
            time.sleep(args.interval)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  DONE — {len(all_snapshots)} successful polls out of {num_polls}")
    if len(all_snapshots) >= 2:
        first = all_snapshots[0]
        last = all_snapshots[-1]
        overall = diff_vehicles(first["vehicles"], last["vehicles"])
        print(f"  Overall changes (poll 1 → {len(all_snapshots)}):")
        print(f"    New trips: {overall['new_trips']}  |  Removed: {overall['removed_trips']}  |  Moved: {len(overall['moved'])}  |  Status changed: {len(overall['status_changed'])}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
