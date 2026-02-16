"""
Arrival event detector for ACE GTFS-RT feed.

Polls the MTA feed, detects train arrivals from `vehicle` entities,
and enriches them with track data from `trip_update.stop_time_update`.

Architecture:
  1. Each poll cycle: parse feed into vehicle + trip_update entities
  2. Build (trip_id, stop_id) -> track lookup from all trip_update.stop_time_update
  3. Merge into a persistent track_cache that survives across polls
  4. Compare vehicle positions to previous poll to detect new arrivals
  5. Emit baseline-schema records for new arrivals, enriched with cached track

A "new arrival" = vehicle is STOPPED_AT a stop_id it wasn't at in the previous poll.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import json

import apache_beam as beam
from apache_beam.coders import PickleCoder
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec

logger = logging.getLogger(__name__)


# ---------- core logic ----------

class ArrivalDetector:
    """Stateful detector that turns raw GTFS-RT polls into arrival events.

    Args:
        route_filter: Set of route_ids to keep (default: A, C, E).
        stop_filter:  Optional set of stop_ids to restrict to.
                      If None, all southbound stops are kept.
    """

    ALLOWED_ROUTES = {"A", "C", "E"}

    def __init__(
        self,
        route_filter: Optional[Set[str]] = None,
        stop_filter: Optional[Set[str]] = None,
    ):
        self.route_filter = route_filter or self.ALLOWED_ROUTES
        self.stop_filter = stop_filter

        # (trip_id, stop_id) -> track string (e.g. "A1", "A3")
        # Persists across polls so tracks aren't lost when stops drop off trip_update
        self.track_cache: Dict[Tuple[str, str], str] = {}

        # Previous poll's vehicle snapshot: trip_id -> {stop_id, status, timestamp}
        self.prev_vehicles: Dict[str, Dict[str, Any]] = {}

    # ---------- public API ----------

    def process_feed(self, feed: dict) -> List[dict]:
        """Process a single feed snapshot. Returns new arrival records.

        Args:
            feed: Parsed JSON feed dict with 'entity' list (from fetch_snapshot
                  format or gtfs_poller format).

        Returns:
            List of baseline-schema dicts for newly detected arrivals.
        """
        entities = feed.get("entity", [])

        # Step 1: separate entity types
        trip_updates = [e for e in entities if "trip_update" in e]
        vehicles = [e for e in entities if "vehicle" in e]

        # Step 2: update track cache + extract trip metadata
        trip_meta = self._update_track_cache(trip_updates)

        # Step 3: build current vehicle snapshot
        curr_vehicles = self._parse_vehicles(vehicles)

        # Step 4: detect new arrivals by diffing against previous snapshot
        arrivals = self._detect_arrivals(curr_vehicles, trip_meta)

        # Step 5: rotate snapshot
        self.prev_vehicles = curr_vehicles

        return arrivals

    # ---------- internals ----------

    def _update_track_cache(self, trip_updates: List[dict]) -> Dict[str, dict]:
        """Extract (trip_id, stop_id) -> track from all stop_time_updates,
        and return trip_id -> {start_date, train_id, direction} metadata.
        """
        trip_meta = {}
        for e in trip_updates:
            tu = e["trip_update"]
            trip = tu["trip"]
            trip_id = trip["trip_id"]

            trip_meta[trip_id] = {
                "start_date": trip.get("start_date"),
                "train_id": trip.get("train_id"),
                "direction": trip.get("direction"),
            }

            for stu in tu.get("stop_time_update", []):
                stop_id = stu.get("stop_id")
                if not stop_id:
                    continue

                track = stu.get("actual_track") or stu.get("scheduled_track")
                if track:
                    self.track_cache[(trip_id, stop_id)] = track

        return trip_meta

    def _parse_vehicles(self, vehicles: List[dict]) -> Dict[str, Dict[str, Any]]:
        """Parse vehicle entities into {trip_id: {...}} snapshot.

        Handles both formats:
          - fetch_snapshot.py: vehicle.trip_id (flat)
          - gtfs_poller.py:   vehicle.trip.trip_id (nested)
        """
        result = {}
        for e in vehicles:
            v = e["vehicle"]

            # Handle flat vs nested trip_id
            if "trip_id" in v:
                trip_id = v["trip_id"]
                route_id = v.get("route_id")
            elif "trip" in v:
                trip_id = v["trip"].get("trip_id")
                route_id = v["trip"].get("route_id")
            else:
                continue

            if not trip_id:
                continue

            result[trip_id] = {
                "stop_id": v.get("stop_id"),
                "current_status": v.get("current_status"),
                "timestamp": v.get("timestamp"),
                "route_id": route_id,
            }

        return result

    def _detect_arrivals(
        self,
        curr_vehicles: Dict[str, Dict[str, Any]],
        trip_meta: Dict[str, dict],
    ) -> List[dict]:
        """Compare current vs previous vehicle snapshot to find new arrivals.

        An arrival is detected when:
          - A vehicle is STOPPED_AT a stop_id, AND
          - In the previous poll it was either: absent, at a different stop,
            or not STOPPED_AT.

        On the very first poll (no previous data), we treat all STOPPED_AT as
        arrivals so we can validate the pipeline immediately.
        """
        arrivals = []
        first_poll = len(self.prev_vehicles) == 0

        for trip_id, veh in curr_vehicles.items():
            stop_id = veh.get("stop_id")
            status = veh.get("current_status")
            route_id = veh.get("route_id")
            timestamp = veh.get("timestamp")

            if not stop_id or status != "STOPPED_AT":
                continue

            # Route filter
            if route_id and route_id not in self.route_filter:
                continue

            # Stop filter (if configured)
            if self.stop_filter and stop_id not in self.stop_filter:
                continue

            # Check if this is a NEW arrival
            prev = self.prev_vehicles.get(trip_id)
            is_new = (
                first_poll
                or prev is None
                or prev.get("stop_id") != stop_id
                or prev.get("current_status") != "STOPPED_AT"
            )

            if not is_new:
                continue

            # --- Build baseline record ---

            track = self.track_cache.get((trip_id, stop_id))
            direction = "S" if stop_id.endswith("S") else "N"

            meta = trip_meta.get(trip_id, {})
            start_date = meta.get("start_date", "")
            trip_uid = f"{start_date}_{trip_id}" if start_date else trip_id

            trip_date = ""
            if start_date and len(start_date) == 8:
                trip_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"

            arrival_time = ""
            if timestamp:
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                arrival_time = dt.isoformat()

            arrivals.append({
                "trip_uid": trip_uid,
                "stop_id": stop_id,
                "track": track,
                "route_id": route_id or "UNKNOWN",
                "direction": direction,
                "arrival_time": arrival_time,
                "trip_date": trip_date,
            })

        return arrivals

    def prune_track_cache(self, active_trip_ids: Set[str], max_age_entries: int = 5000):
        """Remove stale entries from track cache to prevent unbounded growth.

        Call periodically (e.g. every 100 polls).
        Removes entries for trip_ids no longer in the feed.
        """
        if len(self.track_cache) <= max_age_entries:
            return

        stale_keys = [
            k for k in self.track_cache
            if k[0] not in active_trip_ids
        ]
        for k in stale_keys:
            del self.track_cache[k]

        logger.info(
            f"Pruned {len(stale_keys)} stale track cache entries, "
            f"{len(self.track_cache)} remaining"
        )


# ---------- Beam wrapper ----------

class DetectArrivalsFn(beam.DoFn):
    """Stateful DoFn that wraps ArrivalDetector for streaming pipelines.

    Persists track_cache and prev_vehicles in Beam state so they survive
    across messages and worker restarts.  Keyed by a constant so all
    feed snapshots route to the same state partition.

    Input:  ("all_routes", raw_feed_bytes)
    Output: baseline-schema dict per new arrival
    """

    TRACK_CACHE = ReadModifyWriteStateSpec("track_cache", PickleCoder())
    PREV_VEHICLES = ReadModifyWriteStateSpec("prev_vehicles", PickleCoder())

    def process(
        self,
        element,
        track_cache_state=beam.DoFn.StateParam(TRACK_CACHE),
        prev_vehicles_state=beam.DoFn.StateParam(PREV_VEHICLES),
    ):
        _, raw_message = element

        # Parse JSON
        try:
            if isinstance(raw_message, bytes):
                feed = json.loads(raw_message.decode("utf-8"))
            else:
                feed = json.loads(raw_message)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Failed to parse feed message: %s", e)
            return

        # Restore state into a fresh ArrivalDetector
        detector = ArrivalDetector()
        detector.track_cache = track_cache_state.read() or {}
        detector.prev_vehicles = prev_vehicles_state.read() or {}

        # Run the proven detection logic
        arrivals = detector.process_feed(feed)

        # Persist updated state back to Beam
        track_cache_state.write(detector.track_cache)
        prev_vehicles_state.write(detector.prev_vehicles)

        # Yield each arrival as a separate element
        for record in arrivals:
            yield record
