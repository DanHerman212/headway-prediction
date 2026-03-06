"""
Investigate dwell time signals in MTA GTFS-RT feed.

Two potential data sources for dwell time (time at station):
  1. trip_update: does arrival_time != departure_time for any stop?
  2. vehicle diffs: bracket STOPPED_AT duration across polls

Polls every 15s for 3 minutes to observe full dwell cycles.

Usage:
    python realtime_ingestion/investigate_dwell_times.py
    python realtime_ingestion/investigate_dwell_times.py --interval 10 --duration 300
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

URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"
ACE_ROUTES = {"A", "C", "E"}
_STATUS = {0: "INCOMING_AT", 1: "STOPPED_AT", 2: "IN_TRANSIT_TO"}


def fetch_feed():
    try:
        resp = requests.get(URL, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  !! Fetch failed: {e}")
        return None

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)
    now = datetime.now(timezone.utc)

    vehicles = {}
    trip_updates = {}

    for entity in feed.entity:
        if entity.HasField("trip_update"):
            tu = entity.trip_update
            trip_id = tu.trip.trip_id
            route_id = tu.trip.route_id if tu.trip.HasField("route_id") else None
            if route_id not in ACE_ROUTES:
                continue

            stops = []
            for su in tu.stop_time_update:
                sd = {"stop_id": su.stop_id}
                if su.HasField("arrival"):
                    sd["arrival_time"] = su.arrival.time if su.arrival.HasField("time") else None
                if su.HasField("departure"):
                    sd["departure_time"] = su.departure.time if su.departure.HasField("time") else None
                stops.append(sd)
            trip_updates[trip_id] = {"route_id": route_id, "stops": stops}

        if entity.HasField("vehicle"):
            v = entity.vehicle
            trip_id = v.trip.trip_id if v.trip.HasField("trip_id") else None
            route_id = v.trip.route_id if v.trip.HasField("route_id") else None
            if not trip_id or route_id not in ACE_ROUTES:
                continue
            vehicles[trip_id] = {
                "route_id": route_id,
                "stop_id": v.stop_id if v.HasField("stop_id") else None,
                "current_status": _STATUS.get(v.current_status, f"?{v.current_status}"),
                "timestamp": v.timestamp if v.HasField("timestamp") else None,
            }

    return {
        "header_ts": feed.header.timestamp,
        "fetch_time": now,
        "vehicles": vehicles,
        "trip_updates": trip_updates,
    }


def ts_fmt(ts):
    if ts is None:
        return "None"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="Investigate dwell time signals")
    parser.add_argument("--interval", type=int, default=15, help="Poll interval seconds (default: 15)")
    parser.add_argument("--duration", type=int, default=180, help="Total duration seconds (default: 180)")
    args = parser.parse_args()

    num_polls = (args.duration // args.interval) + 1
    print(f"Polling ACE every {args.interval}s for {args.duration}s ({num_polls} polls)")
    print(f"Goal: measure dwell times from vehicle status transitions\n")

    # ---- Question 1: Do trip_update arrival/departure times ever differ? ----
    arr_dep_diffs = []  # (trip_id, stop_id, arrival, departure, delta)

    # ---- Question 2: Track STOPPED_AT durations ----
    # trip_id -> {stop_id, first_seen_stopped_ts, first_seen_poll, vehicle_ts_at_arrival}
    active_dwells = {}
    # Completed dwells: list of measured dwell records
    completed_dwells = []

    for poll_num in range(1, num_polls + 1):
        snap = fetch_feed()
        if snap is None:
            if poll_num < num_polls:
                time.sleep(args.interval)
            continue

        poll_time = snap["fetch_time"]
        header_ts = snap["header_ts"]

        # --- Q1: Check all trip_updates for arr != dep ---
        for trip_id, tu in snap["trip_updates"].items():
            for s in tu["stops"]:
                arr = s.get("arrival_time")
                dep = s.get("departure_time")
                if arr is not None and dep is not None and arr != dep:
                    arr_dep_diffs.append({
                        "trip_id": trip_id,
                        "route_id": tu["route_id"],
                        "stop_id": s["stop_id"],
                        "arrival_time": arr,
                        "departure_time": dep,
                        "delta_s": dep - arr,
                    })

        # --- Q2: Track vehicle STOPPED_AT transitions ---
        curr_vehicles = snap["vehicles"]

        # Check for departures (was in active_dwells, now moved or gone)
        departed = []
        for trip_id, dwell in list(active_dwells.items()):
            cv = curr_vehicles.get(trip_id)
            if cv is None:
                # Trip gone
                departed.append((trip_id, "TRIP_ENDED", None, None))
            elif cv["stop_id"] != dwell["stop_id"]:
                # Moved to different stop
                departed.append((trip_id, cv["current_status"], cv["stop_id"], cv["timestamp"]))
            elif cv["current_status"] != "STOPPED_AT":
                # Same stop but now IN_TRANSIT_TO or INCOMING_AT
                departed.append((trip_id, cv["current_status"], cv["stop_id"], cv["timestamp"]))

        for trip_id, new_status, new_stop, new_ts in departed:
            dwell = active_dwells.pop(trip_id)
            # Observed dwell = time between first seen STOPPED_AT and departure
            departure_poll_time = poll_time
            observed_dwell_s = (departure_poll_time - dwell["first_seen_time"]).total_seconds()

            # Also try to get predicted arrival/departure from trip_update
            # for the stop where the dwell happened
            tu = snap["trip_updates"].get(trip_id)
            pred_arr = pred_dep = None
            if tu:
                for s in tu["stops"]:
                    if s["stop_id"] == dwell["stop_id"]:
                        pred_arr = s.get("arrival_time")
                        pred_dep = s.get("departure_time")
                        break

            completed_dwells.append({
                "trip_id": trip_id,
                "route_id": dwell["route_id"],
                "stop_id": dwell["stop_id"],
                "arrival_vehicle_ts": dwell["vehicle_ts_at_arrival"],
                "first_seen_stopped": dwell["first_seen_time"].isoformat(),
                "departure_poll_time": departure_poll_time.isoformat(),
                "observed_dwell_s": observed_dwell_s,
                "polls_stopped": poll_num - dwell["first_seen_poll"],
                "departure_status": new_status,
                "new_stop": new_stop,
                "pred_arrival": pred_arr,
                "pred_departure": pred_dep,
            })

        # Check for new STOPPED_AT (arrivals)
        for trip_id, cv in curr_vehicles.items():
            if cv["current_status"] != "STOPPED_AT":
                continue
            if trip_id in active_dwells:
                # Already tracking this dwell
                continue
            active_dwells[trip_id] = {
                "stop_id": cv["stop_id"],
                "route_id": cv["route_id"],
                "vehicle_ts_at_arrival": cv["timestamp"],
                "first_seen_time": poll_time,
                "first_seen_poll": poll_num,
            }

        # --- Print poll summary ---
        n_stopped = sum(1 for v in curr_vehicles.values() if v["current_status"] == "STOPPED_AT")
        n_transit = sum(1 for v in curr_vehicles.values() if v["current_status"] == "IN_TRANSIT_TO")
        n_incoming = sum(1 for v in curr_vehicles.values() if v["current_status"] == "INCOMING_AT")

        new_completions = [d for d in completed_dwells if d["departure_poll_time"] == poll_time.isoformat()] if departed else []

        print(f"Poll {poll_num:>2} | {poll_time.strftime('%H:%M:%S')} | "
              f"STOPPED={n_stopped} TRANSIT={n_transit} INCOMING={n_incoming} | "
              f"active_dwells={len(active_dwells)} | "
              f"completed={len(new_completions)} this poll, {len(completed_dwells)} total")

        for d in new_completions:
            print(f"  >> {d['route_id']} {d['trip_id']} @ {d['stop_id']}: "
                  f"dwell={d['observed_dwell_s']:.0f}s ({d['polls_stopped']} polls) "
                  f"→ {d['departure_status']}"
                  f"{' to ' + d['new_stop'] if d['new_stop'] else ''}")

        if poll_num < num_polls:
            time.sleep(args.interval)

    # ============================================================
    # Final analysis
    # ============================================================
    print(f"\n{'='*80}")
    print(f"  DWELL TIME ANALYSIS")
    print(f"{'='*80}")

    # Q1: arrival vs departure in trip_update
    print(f"\n--- Q1: Do trip_update arrival_time and departure_time ever differ? ---")
    if arr_dep_diffs:
        print(f"  YES! Found {len(arr_dep_diffs)} stop predictions with arr != dep")
        print(f"  Sample diffs:")
        for d in arr_dep_diffs[:10]:
            print(f"    {d['route_id']} {d['trip_id']} @ {d['stop_id']}: "
                  f"arr={ts_fmt(d['arrival_time'])} dep={ts_fmt(d['departure_time'])} "
                  f"delta={d['delta_s']}s")
        deltas = [d["delta_s"] for d in arr_dep_diffs]
        print(f"\n  Delta stats: min={min(deltas)}s  max={max(deltas)}s  "
              f"mean={sum(deltas)/len(deltas):.1f}s  count={len(deltas)}")
    else:
        total_stops_checked = sum(
            len(tu["stops"]) for tu in snap["trip_updates"].values()
        ) * num_polls
        print(f"  NO — arrival_time == departure_time for all {total_stops_checked:,} "
              f"stop predictions across {num_polls} polls")
        print(f"  MTA does not provide separate predicted dwell times in trip_update")

    # Q2: Observed dwells from vehicle tracking
    print(f"\n--- Q2: Observed dwell times from vehicle status tracking ---")
    print(f"  Completed dwell measurements: {len(completed_dwells)}")
    print(f"  Still in progress at end: {len(active_dwells)}")

    if completed_dwells:
        dwells_s = [d["observed_dwell_s"] for d in completed_dwells]
        # Bucket by length
        short = [d for d in dwells_s if d <= args.interval]  # only seen in 1 poll
        medium = [d for d in dwells_s if args.interval < d <= 60]
        long_ = [d for d in dwells_s if d > 60]

        print(f"\n  Distribution:")
        print(f"    <= {args.interval}s (1 poll):  {len(short):>3} ({100*len(short)//len(dwells_s)}%)")
        print(f"    {args.interval+1}-60s:          {len(medium):>3} ({100*len(medium)//len(dwells_s)}%)")
        print(f"    > 60s:            {len(long_):>3} ({100*len(long_)//len(dwells_s)}%)")
        print(f"\n  Stats: min={min(dwells_s):.0f}s  max={max(dwells_s):.0f}s  "
              f"mean={sum(dwells_s)/len(dwells_s):.1f}s  median={sorted(dwells_s)[len(dwells_s)//2]:.0f}s")

        # Show some example full dwell cycles
        good_dwells = [d for d in completed_dwells if d["observed_dwell_s"] > args.interval]
        if good_dwells:
            print(f"\n  Example multi-poll dwells (observed > {args.interval}s):")
            for d in good_dwells[:8]:
                print(f"    {d['route_id']} {d['trip_id']} @ {d['stop_id']}: "
                      f"{d['observed_dwell_s']:.0f}s ({d['polls_stopped']} polls) "
                      f"→ {d['departure_status']}")

        print(f"\n  Measurement precision: ±{args.interval}s "
              f"(arrival could be up to {args.interval}s before first STOPPED_AT poll, "
              f"departure up to {args.interval}s before status change detected)")

        # Check: did any have different pred_arrival vs pred_departure?
        with_pred = [d for d in completed_dwells if d["pred_arrival"] and d["pred_departure"]]
        pred_diffs = [d for d in with_pred if d["pred_arrival"] != d["pred_departure"]]
        print(f"\n  Completed dwells with trip_update predictions: {len(with_pred)}")
        print(f"  Of those, pred_arrival != pred_departure: {len(pred_diffs)}")

    print(f"\n{'='*80}")
    print(f"  CONCLUSION")
    print(f"{'='*80}")
    if not arr_dep_diffs:
        print(f"  MTA's trip_update gives identical arrival/departure times per stop.")
        print(f"  Dwell time is NOT available from predictions alone.")
        print(f"")
        print(f"  To compute dwell times, you need to track vehicle status transitions:")
        print(f"    arrival  = first poll where vehicle is STOPPED_AT a stop")
        print(f"    departure = first poll where vehicle left that stop")
        print(f"    dwell    = departure_poll_time - arrival_poll_time")
        print(f"")
        print(f"  Precision depends on poll interval ({args.interval}s → ±{args.interval}s error).")
        print(f"  A tighter interval (e.g. 10s) gives better resolution but more API calls.")
    else:
        print(f"  MTA provides separate arrival/departure times in trip_update for some stops.")
        print(f"  These can be used directly as predicted dwell = dep - arr.")
    print()


if __name__ == "__main__":
    main()
