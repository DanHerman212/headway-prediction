"""
Investigate arrival & departure signals in MTA GTFS-RT feed.

Polls the ACE feed every 15s for 2 minutes and tracks:
  1. Predicted arrival/departure times from trip_update.stop_time_update
  2. Actual arrival events (vehicle newly STOPPED_AT a stop)
  3. Actual departure events (vehicle was STOPPED_AT, now at next stop)

Usage:
    python realtime_ingestion/inspect_arrivals_departures.py
    python realtime_ingestion/inspect_arrivals_departures.py --interval 15 --duration 120
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
# HasField('current_status') returns False for IN_TRANSIT_TO (enum=2) in some
# protobuf versions, so we map the raw int directly.
_VEHICLE_STATUS_NAMES = {0: "INCOMING_AT", 1: "STOPPED_AT", 2: "IN_TRANSIT_TO"}

URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"
ACE_ROUTES = {"A", "C", "E"}


def fetch_feed() -> dict | None:
    try:
        resp = requests.get(URL, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  !! Fetch failed: {e}")
        return None

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)

    result = {
        "header_ts": feed.header.timestamp,
        "fetch_time": datetime.now(timezone.utc),
        "trip_updates": {},  # trip_id -> {trip info, stop_time_updates}
        "vehicles": {},  # trip_id -> {vehicle info}
    }

    for entity in feed.entity:
        if entity.HasField("trip_update"):
            tu = entity.trip_update
            trip_id = tu.trip.trip_id
            route_id = tu.trip.route_id if tu.trip.HasField("route_id") else None

            if route_id not in ACE_ROUTES:
                continue

            direction = None
            train_id = None
            if tu.trip.HasExtension(nyct_subway_pb2.nyct_trip_descriptor):
                nyct = tu.trip.Extensions[nyct_subway_pb2.nyct_trip_descriptor]
                train_id = nyct.train_id if nyct.HasField("train_id") else None
                direction = nyct.Direction.Name(nyct.direction) if nyct.HasField("direction") else None

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

            result["trip_updates"][trip_id] = {
                "route_id": route_id,
                "train_id": train_id,
                "direction": direction,
                "stop_time_updates": stops,
            }

        elif entity.HasField("vehicle"):
            v = entity.vehicle
            trip_id = v.trip.trip_id if v.trip.HasField("trip_id") else None
            route_id = v.trip.route_id if v.trip.HasField("route_id") else None

            if not trip_id or route_id not in ACE_ROUTES:
                continue

            result["vehicles"][trip_id] = {
                "route_id": route_id,
                "stop_id": v.stop_id if v.HasField("stop_id") else None,
                "current_status": _VEHICLE_STATUS_NAMES.get(v.current_status, f"UNKNOWN({v.current_status})"),
                "timestamp": v.timestamp if v.HasField("timestamp") else None,
            }

    return result


def ts_str(ts):
    """Format unix timestamp to HH:MM:SS."""
    if ts is None:
        return "None"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


def detect_events(prev_snap, curr_snap, feed_ts):
    """Detect arrival and departure events from vehicle diffs.

    Returns (arrivals, departures) where each is a list of event dicts.
    """
    if prev_snap is None:
        return [], []

    prev_veh = prev_snap["vehicles"]
    curr_veh = curr_snap["vehicles"]
    curr_tu = curr_snap["trip_updates"]

    arrivals = []
    departures = []

    # --- Arrivals: vehicle is newly STOPPED_AT a stop ---
    for trip_id, cv in curr_veh.items():
        stop_id = cv["stop_id"]
        status = cv["current_status"]
        if not stop_id or status != "STOPPED_AT":
            continue

        pv = prev_veh.get(trip_id)
        is_new_arrival = (
            pv is None
            or pv.get("stop_id") != stop_id
            or pv.get("current_status") != "STOPPED_AT"
        )
        if not is_new_arrival:
            continue

        # Look up predicted arrival time for this stop from trip_update
        predicted_arrival = None
        predicted_departure = None
        tu = curr_tu.get(trip_id)
        if tu:
            for stu in tu["stop_time_updates"]:
                if stu["stop_id"] == stop_id:
                    predicted_arrival = stu.get("arrival_time")
                    predicted_departure = stu.get("departure_time")
                    break

        arrivals.append({
            "trip_id": trip_id,
            "route_id": cv["route_id"],
            "stop_id": stop_id,
            "vehicle_ts": cv["timestamp"],
            "feed_ts": feed_ts,
            "prev_stop": pv["stop_id"] if pv else None,
            "prev_status": pv["current_status"] if pv else None,
            "predicted_arrival": predicted_arrival,
            "predicted_departure": predicted_departure,
        })

    # --- Departures: vehicle was STOPPED_AT, now at a different stop or IN_TRANSIT ---
    for trip_id, pv in prev_veh.items():
        if pv.get("current_status") != "STOPPED_AT":
            continue

        cv = curr_veh.get(trip_id)
        if cv is None:
            # Trip disappeared (could be end of line)
            departures.append({
                "trip_id": trip_id,
                "route_id": pv["route_id"],
                "departed_stop": pv["stop_id"],
                "feed_ts": feed_ts,
                "vehicle_ts": None,
                "new_stop": None,
                "new_status": "TRIP_ENDED",
            })
            continue

        departed = (
            cv["stop_id"] != pv["stop_id"]
            or cv["current_status"] in ("IN_TRANSIT_TO", "INCOMING_AT")
        )
        if departed:
            departures.append({
                "trip_id": trip_id,
                "route_id": cv["route_id"],
                "departed_stop": pv["stop_id"],
                "feed_ts": feed_ts,
                "vehicle_ts": cv["timestamp"],
                "new_stop": cv["stop_id"],
                "new_status": cv["current_status"],
            })

    return arrivals, departures


def show_prediction_evolution(all_snapshots, sample_trip_id):
    """Show how predicted arrival times for a trip's stops change across polls."""
    print(f"\n{'='*80}")
    print(f"  PREDICTION EVOLUTION for trip: {sample_trip_id}")
    print(f"{'='*80}")

    # Collect all stops mentioned across all polls for this trip
    all_stops = []
    seen = set()
    for snap in all_snapshots:
        tu = snap["trip_updates"].get(sample_trip_id)
        if tu:
            for stu in tu["stop_time_updates"]:
                sid = stu["stop_id"]
                if sid not in seen:
                    all_stops.append(sid)
                    seen.add(sid)

    if not all_stops:
        print("  (trip not found in trip_updates)")
        return

    # Show table: stop_id | predicted arrival at each poll
    header = f"  {'Stop':<8}"
    for i, snap in enumerate(all_snapshots, 1):
        header += f" | Poll {i:>2} arr"
    print(header)
    print(f"  {'-'*len(header)}")

    for stop_id in all_stops[:15]:  # limit rows
        row = f"  {stop_id:<8}"
        for snap in all_snapshots:
            tu = snap["trip_updates"].get(sample_trip_id)
            arr = None
            if tu:
                for stu in tu["stop_time_updates"]:
                    if stu["stop_id"] == stop_id:
                        arr = stu.get("arrival_time")
                        break
            row += f" | {ts_str(arr):>11}"
        print(row)

    # Also show where the vehicle was at each poll
    vrow = f"  {'[veh@]':<8}"
    for snap in all_snapshots:
        v = snap["vehicles"].get(sample_trip_id)
        if v:
            vrow += f" | {(v['stop_id'] or '?'):>11}"
        else:
            vrow += f" | {'—':>11}"
    print(vrow)


def main():
    parser = argparse.ArgumentParser(description="Inspect arrival/departure signals")
    parser.add_argument("--interval", type=int, default=15, help="seconds between polls (default: 15)")
    parser.add_argument("--duration", type=int, default=120, help="total seconds to run (default: 120)")
    args = parser.parse_args()

    num_polls = (args.duration // args.interval) + 1
    print(f"Polling ACE feed every {args.interval}s for {args.duration}s ({num_polls} polls)")
    print(f"Tracking: predicted arrivals/departures from trip_update")
    print(f"          + actual arrival/departure events from vehicle diffs\n")

    all_snapshots = []
    prev_snap = None
    all_arrivals = []
    all_departures = []

    for i in range(1, num_polls + 1):
        snap = fetch_feed()
        if snap is None:
            print(f"  Poll #{i} FAILED")
            if i < num_polls:
                time.sleep(args.interval)
            continue

        all_snapshots.append(snap)
        n_tu = len(snap["trip_updates"])
        n_v = len(snap["vehicles"])

        arrivals, departures = detect_events(prev_snap, snap, snap["header_ts"])

        print(f"{'='*80}")
        print(f"  POLL #{i}  |  {snap['fetch_time'].strftime('%H:%M:%S')} UTC  |  Feed ts: {snap['header_ts']}")
        print(f"  ACE vehicles: {n_v}  |  ACE trip_updates: {n_tu}")

        if prev_snap is None:
            print(f"  (first poll — no diff yet)")

            # Show a sample trip_update to illustrate prediction data
            sample_trips = [
                tid for tid, tu in snap["trip_updates"].items()
                if tu["route_id"] == "A" and len(tu["stop_time_updates"]) >= 3
            ]
            if sample_trips:
                tid = sample_trips[0]
                tu = snap["trip_updates"][tid]
                v = snap["vehicles"].get(tid, {})
                print(f"\n  --- Sample trip_update: {tid} (route {tu['route_id']}, dir={tu['direction']}) ---")
                print(f"  Vehicle currently at: {v.get('stop_id', '?')} ({v.get('current_status', '?')})")
                print(f"  Predicted stop times (next {min(6, len(tu['stop_time_updates']))} stops):")
                for stu in tu["stop_time_updates"][:6]:
                    arr = ts_str(stu.get("arrival_time"))
                    dep = ts_str(stu.get("departure_time"))
                    trk = stu.get("actual_track") or stu.get("scheduled_track") or "?"
                    print(f"    {stu['stop_id']}  arr={arr}  dep={dep}  track={trk}")
        else:
            print(f"\n  ARRIVALS detected: {len(arrivals)}")
            for a in arrivals:
                pred_info = ""
                if a["predicted_arrival"]:
                    pred_info = f"  predicted_arr={ts_str(a['predicted_arrival'])}"
                if a["predicted_departure"]:
                    pred_info += f"  predicted_dep={ts_str(a['predicted_departure'])}"
                print(
                    f"    {a['route_id']} {a['trip_id']}: arrived at {a['stop_id']} "
                    f"(from {a['prev_stop'] or 'new'}, veh_ts={ts_str(a['vehicle_ts'])})"
                    f"{pred_info}"
                )

            print(f"\n  DEPARTURES detected: {len(departures)}")
            for d in departures:
                print(
                    f"    {d['route_id']} {d['trip_id']}: departed {d['departed_stop']} "
                    f"→ {d['new_stop'] or '?'} ({d['new_status']}, veh_ts={ts_str(d['vehicle_ts'])})"
                )

            all_arrivals.extend(arrivals)
            all_departures.extend(departures)

        prev_snap = snap

        if i < num_polls:
            print(f"\n  Waiting {args.interval}s...")
            time.sleep(args.interval)

    # --- Final summary ---
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Polls: {len(all_snapshots)}  |  Total arrivals: {len(all_arrivals)}  |  Total departures: {len(all_departures)}")

    if all_arrivals:
        # Count how many had predictions available
        with_pred = sum(1 for a in all_arrivals if a["predicted_arrival"] is not None)
        print(f"\n  Arrivals with predicted_arrival available: {with_pred}/{len(all_arrivals)} ({100*with_pred//len(all_arrivals)}%)")

        # Show prediction accuracy for those that had it
        errors = []
        for a in all_arrivals:
            if a["predicted_arrival"] and a["vehicle_ts"]:
                err = a["vehicle_ts"] - a["predicted_arrival"]
                errors.append(err)
        if errors:
            avg_err = sum(errors) / len(errors)
            print(f"  Prediction error (vehicle_ts - predicted_arrival):")
            print(f"    Mean: {avg_err:+.0f}s  |  Min: {min(errors):+.0f}s  |  Max: {max(errors):+.0f}s")
            print(f"    (positive = arrived later than predicted)")

    if all_departures:
        by_type = defaultdict(int)
        for d in all_departures:
            by_type[d["new_status"]] += 1
        print(f"\n  Departure types: {dict(by_type)}")

    # --- Prediction evolution for one sample trip ---
    if len(all_snapshots) >= 3:
        # Pick a trip that appeared in most snapshots
        trip_counts = defaultdict(int)
        for snap in all_snapshots:
            for tid in snap["trip_updates"]:
                trip_counts[tid] += 1
        longest = max(trip_counts, key=trip_counts.get)
        show_prediction_evolution(all_snapshots, longest)

    print()


if __name__ == "__main__":
    main()
