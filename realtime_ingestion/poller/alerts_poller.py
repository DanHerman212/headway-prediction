"""
Service Alerts Poller for NYC Subway

Fetches Service Alerts JSON data from MTA and publishes to Pub/Sub.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from google.cloud import pubsub_v1

from config import Config, get_config

logger = logging.getLogger(__name__)


class AlertsPoller:
    """Fetches Service Alerts JSON feed and publishes to Pub/Sub."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.config.alerts_topic
        self._session = requests.Session()

    def fetch_feed(self) -> Optional[dict]:
        """Fetch the Service Alerts JSON feed from MTA.
        
        Returns:
            Parsed JSON as dict, or None if fetch failed.
        """
        for attempt in range(self.config.max_retries):
            try:
                response = self._session.get(
                    self.config.alerts_url,
                    timeout=self.config.request_timeout_seconds,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Fetch failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_backoff_seconds * (2 ** attempt))
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                return None
        return None

    def publish(self, feed_data: dict) -> bool:
        """Publish JSON feed data to Pub/Sub.
        
        Args:
            feed_data: Parsed JSON feed dictionary
            
        Returns:
            True if published successfully, False otherwise.
        """
        try:
            # Add ingest timestamp
            feed_data["_ingest_time"] = datetime.now(timezone.utc).isoformat()
            
            json_bytes = json.dumps(feed_data).encode("utf-8")
            future = self.publisher.publish(
                self.topic_path,
                json_bytes,
                feed_type="service-alerts",
            )
            future.result(timeout=30)
            return True
        except Exception as e:
            logger.error(f"Failed to publish: {e}")
            return False

    def poll_once(self) -> dict:
        """Perform a single poll cycle.
        
        Returns:
            Dict with success status and metrics.
        """
        feed_data = self.fetch_feed()
        if feed_data is None:
            return {"success": False, "bytes": 0, "alerts": 0}

        alert_count = len(feed_data.get("entity", []))
        published = self.publish(feed_data)
        
        return {
            "success": published,
            "bytes": len(json.dumps(feed_data)),
            "alerts": alert_count,
        }

    def run(self):
        """Run the polling loop continuously."""
        logger.info(f"Starting alerts poller, interval: {self.config.poll_interval_seconds}s")
        while True:
            start_time = time.time()
            try:
                result = self.poll_once()
                if result["success"]:
                    logger.info(f"Published {result['alerts']} alerts ({result['bytes']} bytes)")
                else:
                    logger.warning("Poll cycle failed")
            except Exception as e:
                logger.error(f"Unexpected error in poll cycle: {e}")

            # Sleep for remaining interval time
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config.poll_interval_seconds - elapsed)
            time.sleep(sleep_time)


def main():
    """Entry point for alerts poller."""
    poller = AlertsPoller()
    poller.run()


if __name__ == "__main__":
    main()
