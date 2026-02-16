"""Analyze vehicle entities and their join to trip_update for track info."""
import json
from collections import Counter

with open("realtime_ingestion/ace_snapshot.json") as f:
    data = json.load(f)

vehs = [e for e in data["entity"] if "vehicle" in e]
tus = [e for e in data["entity"] if "trip_update" in e]

# --- Vehicle schema ---
print("=== VEHICLE SCHEMA ===")
print(f"Count: {len(vehs)}")
print(f"Keys: {list(vehs[0]['vehicle'].keys())}")
statuses = Counter(v["vehicle"]["current_status"] for v in vehs)
print(f"Statuses: {dict(statuses)}")
routes = Counter(v["vehicle"]["route_id"] for v in vehs)
print(f"Routes: {dict(routes)}")
print()

# --- Trip_update schema ---
print("=== TRIP_UPDATE SCHEMA ===")
print(f"Count: {len(tus)}")
print(f"Trip keys: {list(tus[0]['trip_update']['trip'].keys())}")
print(f"STU keys: {list(tus[0]['trip_update']['stop_time_update'][0].keys())}")
print()

# --- Join on trip_id ---
veh_trips = {v["vehicle"]["trip_id"] for v in vehs}
tu_trips = {e["trip_update"]["trip"]["trip_id"] for e in tus}
print("=== TRIP_ID JOIN ===")
print(f"Vehicle trip_ids: {len(veh_trips)}")
print(f"Trip_update trip_ids: {len(tu_trips)}")
print(f"Intersection (1:1 match): {len(veh_trips & tu_trips)}")
print(f"Vehicle-only: {len(veh_trips - tu_trips)}")
print(f"TU-only: {len(tu_trips - veh_trips)}")
print()

# --- Build a lookup: (trip_id, stop_id) -> track ---
tu_lookup = {}
for e in tus:
    tid = e["trip_update"]["trip"]["trip_id"]
    for stu in e["trip_update"]["stop_time_update"]:
        key = (tid, stu["stop_id"])
        tu_lookup[key] = {
            "scheduled_track": stu.get("scheduled_track"),
            "actual_track": stu.get("actual_track"),
        }

# --- Join example: first 5 vehicles ---
print("=== JOIN EXAMPLES (first 5 vehicles) ===")
for v in vehs[:5]:
    vd = v["vehicle"]
    tid = vd["trip_id"]
    sid = vd["stop_id"]
    track_info = tu_lookup.get((tid, sid), {})
    track = track_info.get("actual_track") or track_info.get("scheduled_track", "MISSING")
    print(f"  trip={tid}  stop={sid}  status={vd['current_status']}  ts={vd['timestamp']}  -> track={track}")
print()

# --- How many vehicles can we resolve track for? ---
matched = 0
unmatched_examples = []
for v in vehs:
    vd = v["vehicle"]
    key = (vd["trip_id"], vd["stop_id"])
    if key in tu_lookup:
        matched += 1
    else:
        if len(unmatched_examples) < 5:
            unmatched_examples.append(key)

print(f"=== TRACK RESOLUTION ===")
print(f"Vehicles with track match: {matched}/{len(vehs)}")
print(f"Unmatched examples: {unmatched_examples}")
print()

# --- Corridor vehicles (A28S-A32S) ---
corridor = {"A28S", "A29S", "A30S", "A31S", "A32S"}
print("=== CORRIDOR VEHICLES (A28S-A32S) ===")
for v in vehs:
    vd = v["vehicle"]
    if vd["stop_id"] in corridor:
        tid = vd["trip_id"]
        sid = vd["stop_id"]
        track_info = tu_lookup.get((tid, sid), {})
        track = track_info.get("actual_track") or track_info.get("scheduled_track", "MISSING")
        print(f"  route={vd['route_id']}  trip={tid}  stop={sid}  status={vd['current_status']}  ts={vd['timestamp']}  track={track}")
