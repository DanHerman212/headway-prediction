"""
GTFS-RT Poller for NYC Subway

Fetches GTFS-RT protobuf data from MTA, extracts track information
from MTA extensions, converts to JSON, and publishes to Pub/Sub.

Supports multiple feeds (ACE, BDFM, etc.) via feed_url and feed_id parameters.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import requests
from google.cloud import pubsub_v1

# Import compiled protobuf modules with MTA extensions
import gtfs_realtime_pb2
import nyct_subway_pb2

from config import Config, get_config

logger = logging.getLogger(__name__)


class GTFSPoller:
    """Fetches GTFS-RT feed, converts to JSON, and publishes to Pub/Sub.
    
    Args:
        feed_url: URL to fetch GTFS-RT protobuf from
        topic_path: Full Pub/Sub topic path to publish to
        feed_id: Identifier for this feed (e.g., 'ace', 'bdfm')
        config: Optional Config instance (uses global config if not provided)
    """

    def __init__(
        self,
        feed_url: str,
        topic_path: str,
        feed_id: str,
        config: Optional[Config] = None,
    ):
        self.config = config or get_config()
        self.feed_url = feed_url
        self.topic_path = topic_path
        self.feed_id = feed_id
        self.publisher = pubsub_v1.PublisherClient()
        self._session = requests.Session()

    def fetch_feed(self) -> Optional[bytes]:
        """Fetch the raw GTFS-RT feed from MTA."""
        for attempt in range(self.config.max_retries):
            try:
                response = self._session.get(
                    self.feed_url,
                    timeout=self.config.request_timeout_seconds,
                )
                response.raise_for_status()
                return response.content
            except requests.exceptions.RequestException as e:
                logger.warning(f"[{self.feed_id}] Fetch failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_backoff_seconds * (2 ** attempt))
        return None

    def protobuf_to_json(self, data: bytes) -> Optional[str]:
        """Convert GTFS-RT protobuf to JSON, extracting MTA extension fields."""
        try:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(data)
            
            # Build JSON structure manually to include extension fields
            feed_dict = {
                "header": {
                    "gtfs_realtime_version": feed.header.gtfs_realtime_version,
                    "timestamp": feed.header.timestamp,
                },
                "entity": [],
                "_ingest_time": datetime.now(timezone.utc).isoformat()
            }
            
            # Process each entity
            for entity in feed.entity:
                entity_dict = {"id": entity.id}
                
                # Handle trip updates
                if entity.HasField('trip_update'):
                    trip_update = entity.trip_update
                    trip_dict = {
                        "trip_id": trip_update.trip.trip_id,
                        "route_id": trip_update.trip.route_id if trip_update.trip.HasField('route_id') else None,
                        "start_time": trip_update.trip.start_time if trip_update.trip.HasField('start_time') else None,
                        "start_date": trip_update.trip.start_date if trip_update.trip.HasField('start_date') else None,
                    }
                    
                    # Extract NYCT trip extension
                    if trip_update.trip.HasExtension(nyct_subway_pb2.nyct_trip_descriptor):
                        nyct_trip = trip_update.trip.Extensions[nyct_subway_pb2.nyct_trip_descriptor]
                        trip_dict["train_id"] = nyct_trip.train_id if nyct_trip.HasField('train_id') else None
                        trip_dict["direction"] = nyct_trip.Direction.Name(nyct_trip.direction) if nyct_trip.HasField('direction') else None
                    
                    # Process stop time updates
                    stop_time_updates = []
                    for stop_update in trip_update.stop_time_update:
                        stop_dict = {
                            "stop_id": stop_update.stop_id,
                            "stop_sequence": stop_update.stop_sequence if stop_update.HasField('stop_sequence') else None,
                        }
                        
                        # Add arrival/departure times
                        if stop_update.HasField('arrival'):
                            stop_dict["arrival"] = {
                                "time": stop_update.arrival.time if stop_update.arrival.HasField('time') else None
                            }
                        if stop_update.HasField('departure'):
                            stop_dict["departure"] = {
                                "time": stop_update.departure.time if stop_update.departure.HasField('time') else None
                            }
                        
                        # Extract NYCT stop time extension (track data)
                        if stop_update.HasExtension(nyct_subway_pb2.nyct_stop_time_update):
                            nyct_stop = stop_update.Extensions[nyct_subway_pb2.nyct_stop_time_update]
                            stop_dict["scheduled_track"] = nyct_stop.scheduled_track if nyct_stop.HasField('scheduled_track') else None
                            stop_dict["actual_track"] = nyct_stop.actual_track if nyct_stop.HasField('actual_track') else None
                        
                        stop_time_updates.append(stop_dict)
                    
                    # Build trip_update structure with stop_time_update at correct level
                    entity_dict["trip_update"] = {
                        "trip": trip_dict,
                        "stop_time_update": stop_time_updates
                    }
                
                # Handle vehicle positions (if present)
                elif entity.HasField('vehicle'):
                    vehicle = entity.vehicle
                    vehicle_dict = {
                        "trip": {
                            "trip_id": vehicle.trip.trip_id if vehicle.trip.HasField('trip_id') else None,
                            "route_id": vehicle.trip.route_id if vehicle.trip.HasField('route_id') else None,
                        },
                        "current_stop_sequence": vehicle.current_stop_sequence if vehicle.HasField('current_stop_sequence') else None,
                        "stop_id": vehicle.stop_id if vehicle.HasField('stop_id') else None,
                        "current_status": vehicle.VehicleStopStatus.Name(vehicle.current_status) if vehicle.HasField('current_status') else None,
                        "timestamp": vehicle.timestamp if vehicle.HasField('timestamp') else None,
                    }
                    entity_dict["vehicle"] = vehicle_dict
                
                feed_dict["entity"].append(entity_dict)
            
            return json.dumps(feed_dict)
            
        except Exception as e:
            logger.error(f"Failed to parse protobuf: {e}", exc_info=True)
            return None

    def publish(self, json_data: str) -> bool:
        """Publish JSON feed data to Pub/Sub."""
        try:
            future = self.publisher.publish(
                self.topic_path,
                json_data.encode("utf-8"),
                feed_type="gtfs-rt",
                feed_id=self.feed_id,
            )
            future.result(timeout=30)
            return True
        except Exception as e:
            logger.error(f"[{self.feed_id}] Failed to publish: {e}")
            return False

    def poll_once(self) -> dict:
        """Perform a single poll cycle."""
        raw_data = self.fetch_feed()
        if raw_data is None:
            return {"success": False, "bytes": 0, "entities": 0}

        json_data = self.protobuf_to_json(raw_data)
        if json_data is None:
            return {"success": False, "bytes": 0, "entities": 0}

        try:
            entity_count = len(json.loads(json_data).get("entity", []))
        except:
            entity_count = 0

        published = self.publish(json_data)
        return {"success": published, "bytes": len(json_data), "entities": entity_count}

    def run(self):
        """Run the polling loop continuously."""
        logger.info(f"Starting GTFS poller, interval: {self.config.poll_interval_seconds}s")
        while True:
            start_time = time.time()
            try:
                result = self.poll_once()
                if result["success"]:
                    logger.info(f"Published: {result['entities']} entities, {result['bytes']} bytes")
                else:
                    logger.warning("Poll failed")
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config.poll_interval_seconds - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    GTFSPoller().run()


if __name__ == "__main__":
    main()
