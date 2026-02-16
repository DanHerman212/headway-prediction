"""Analyze the ACE GTFS-RT snapshot to understand the raw schema."""
import json
from collections import defaultdict

with open("realtime_ingestion/ace_snapshot.json") as f:
    data = json.load(f)

all_entities = data["entity"]
# Split by type
trip_updates = [e for e in all_entities if "trip_update" in e]
vehicle_positions = [e for e in all_entities if "vehicle" in e]
print(f"Feed timestamp: {data['header']['timestamp']}")
print(f"Total entities: {len(all_entities)}  (trip_update: {len(trip_updates)}, vehicle: {len(vehicle_positions)})")
print()

entities = trip_updates  # only work with trip_update entities below

# --- Routes and directions ---
routes = set()
directions = set()
for e in entities:
    trip = e["trip_update"]["trip"]
    routes.add(trip["route_id"])
    directions.add(trip.get("direction"))

print(f"Routes: {sorted(routes)}")
print(f"Directions: {sorted(directions, key=str)}")
print()

# --- Trip-level fields ---
print("=== TRIP-LEVEL FIELDS (first entity) ===")
trip0 = entities[0]["trip_update"]["trip"]
for k, v in trip0.items():
    print(f"  {k}: {v!r}")
print()

# --- Stop-time-update fields ---
print("=== STOP_TIME_UPDATE FIELDS (first stop of first entity) ===")
stu0 = entities[0]["trip_update"]["stop_time_update"][0]
for k, v in stu0.items():
    print(f"  {k}: {v!r}")
print()

# --- Target corridor stops (8th Ave, A25-A36) ---
target_stops = defaultdict(set)  # stop_id -> set of (scheduled_track, actual_track)
for e in entities:
    for stu in e["trip_update"]["stop_time_update"]:
        sid = stu["stop_id"]
        if sid.startswith("A") and len(sid) >= 3 and sid[1:3].isdigit():
            num = int(sid[1:3])
            if 25 <= num <= 36:
                sched = stu.get("scheduled_track", "N/A")
                actual = stu.get("actual_track", "N/A")
                target_stops[sid].add((sched, actual))

print("=== RELEVANT STOPS & TRACK VALUES ===")
for sid in sorted(target_stops.keys()):
    tracks = target_stops[sid]
    print(f"  {sid}: {sorted(tracks)}")
print()

# --- Example: a trip that passes through A32S (W 4th St) ---
print("=== EXAMPLE TRIP PASSING THROUGH A32S (W 4th St) ===")
for e in entities:
    stops = e["trip_update"]["stop_time_update"]
    stop_ids = [s["stop_id"] for s in stops]
    if "A32S" in stop_ids:
        trip = e["trip_update"]["trip"]
        print(f"  trip_id: {trip['trip_id']}")
        print(f"  route_id: {trip['route_id']}")
        print(f"  direction: {trip.get('direction')}")
        print(f"  train_id: {trip.get('train_id')}")
        print(f"  start_time: {trip['start_time']}")
        print(f"  start_date: {trip['start_date']}")
        print()
        print("  Stop sequence:")
        for s in stops:
            print(f"    {s['stop_id']:6s}  arr={s['arrival_time']}  sched_track={s.get('scheduled_track','N/A'):4s}  actual_track={s.get('actual_track','N/A'):4s}")
        break
print()

# --- How many trips hit each target stop ---
print("=== TRIP COUNTS PER TARGET STOP ===")
stop_trip_count = defaultdict(int)
for e in entities:
    seen = set()
    for stu in e["trip_update"]["stop_time_update"]:
        sid = stu["stop_id"]
        if sid in target_stops and sid not in seen:
            stop_trip_count[sid] += 1
            seen.add(sid)

for sid in sorted(stop_trip_count.keys()):
    print(f"  {sid}: {stop_trip_count[sid]} trips")
