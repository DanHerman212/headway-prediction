"""Fetch a single ACE GTFS-RT snapshot from MTA and dump to JSON."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "poller"))

import requests
import gtfs_realtime_pb2
import nyct_subway_pb2
from datetime import datetime, timezone

URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"

def main():
    print("Fetching ACE feed...")
    resp = requests.get(URL, timeout=10)
    resp.raise_for_status()
    raw = resp.content
    print(f"Raw bytes: {len(raw)}")

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(raw)
    print(f"Header timestamp: {feed.header.timestamp}")
    print(f"Entity count: {len(feed.entity)}")

    feed_dict = {
        "header": {
            "gtfs_realtime_version": feed.header.gtfs_realtime_version,
            "timestamp": feed.header.timestamp,
        },
        "entity": [],
        "_fetch_time": datetime.now(timezone.utc).isoformat(),
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
                    if nyct.HasField("direction")
                    else None
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
                    sd["scheduled_track"] = (
                        ns.scheduled_track if ns.HasField("scheduled_track") else None
                    )
                    sd["actual_track"] = (
                        ns.actual_track if ns.HasField("actual_track") else None
                    )
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
                    if v.HasField("current_status")
                    else None
                ),
                "timestamp": v.timestamp if v.HasField("timestamp") else None,
            }

        feed_dict["entity"].append(ed)

    # Write full snapshot
    outpath = os.path.join(os.path.dirname(__file__), "ace_snapshot.json")
    with open(outpath, "w") as f:
        json.dump(feed_dict, f, indent=2)

    # Summary
    routes = set()
    tu_count = vp_count = 0
    for e in feed_dict["entity"]:
        if "trip_update" in e:
            tu_count += 1
            r = e["trip_update"]["trip"].get("route_id")
            if r:
                routes.add(r)
        if "vehicle" in e:
            vp_count += 1

    print(f"\nSummary:")
    print(f"  Trip updates: {tu_count}")
    print(f"  Vehicle positions: {vp_count}")
    print(f"  Routes: {sorted(routes)}")
    print(f"  Wrote to: {outpath}")

    # Show 2 sample trip_updates
    samples = [e for e in feed_dict["entity"] if "trip_update" in e][:2]
    print(f"\n--- Sample entities (first 2 trip_updates) ---")
    print(json.dumps(samples, indent=2))


if __name__ == "__main__":
    main()
