"""
Main entry point for GTFS and Alerts Pollers

Runs both pollers concurrently using threading.
Both feeds are polled from the same process/VM.
"""

import logging
import signal
import sys
import threading
import time

from gtfs_poller import GTFSPoller
from alerts_poller import AlertsPoller
from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Handle graceful shutdown on SIGTERM/SIGINT."""
    
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True


def run_poller(poller, name: str, shutdown: GracefulShutdown):
    """Generic poller thread function.
    
    Args:
        poller: Poller instance (GTFSPoller or AlertsPoller)
        name: Name for logging
        shutdown: Shutdown handler
    """
    config = get_config()
    logger.info(f"Starting {name} poller thread")
    
    while not shutdown.shutdown_requested:
        start_time = time.time()
        
        try:
            result = poller.poll_once()
            if result["success"]:
                # GTFS has 'entities', Alerts has 'alerts'
                count = result.get("entities", result.get("alerts", 0))
                logger.info(f"[{name}] Published {count} items, {result['bytes']} bytes")
            else:
                logger.warning(f"[{name}] Poll failed")
        except Exception as e:
            logger.error(f"[{name}] Error: {e}", exc_info=True)
        
        # Sleep in small increments to check for shutdown
        elapsed = time.time() - start_time
        sleep_time = max(0, config.poll_interval_seconds - elapsed)
        
        while sleep_time > 0 and not shutdown.shutdown_requested:
            time.sleep(min(1, sleep_time))
            sleep_time -= 1
    
    logger.info(f"{name} poller thread stopped")


def main():
    """Main entry point."""
    config = get_config()
    shutdown = GracefulShutdown()
    
    logger.info("=" * 50)
    logger.info("NYC Subway Feed Poller")
    logger.info("=" * 50)
    logger.info(f"Project: {config.project_id}")
    logger.info(f"GTFS ACE Topic: {config.gtfs_ace_topic}")
    logger.info(f"GTFS BDFM Topic: {config.gtfs_bdfm_topic}")
    logger.info(f"Alerts Topic: {config.alerts_topic}")
    logger.info(f"Poll Interval: {config.poll_interval_seconds}s")
    logger.info("=" * 50)
    
    # Create pollers
    gtfs_ace_poller = GTFSPoller(
        feed_url=config.gtfs_ace_url,
        topic_path=config.gtfs_ace_topic,
        feed_id="ace",
        config=config,
    )
    gtfs_bdfm_poller = GTFSPoller(
        feed_url=config.gtfs_bdfm_url,
        topic_path=config.gtfs_bdfm_topic,
        feed_id="bdfm",
        config=config,
    )
    alerts_poller = AlertsPoller(config)
    
    # Start poller threads
    gtfs_ace_thread = threading.Thread(
        target=run_poller,
        args=(gtfs_ace_poller, "GTFS-ACE", shutdown),
        daemon=True,
    )
    gtfs_bdfm_thread = threading.Thread(
        target=run_poller,
        args=(gtfs_bdfm_poller, "GTFS-BDFM", shutdown),
        daemon=True,
    )
    alerts_thread = threading.Thread(
        target=run_poller,
        args=(alerts_poller, "Alerts", shutdown),
        daemon=True,
    )
    
    gtfs_ace_thread.start()
    gtfs_bdfm_thread.start()
    alerts_thread.start()
    
    # Wait for shutdown
    try:
        while not shutdown.shutdown_requested:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    logger.info("Waiting for threads to finish...")
    gtfs_ace_thread.join(timeout=5)
    gtfs_bdfm_thread.join(timeout=5)
    alerts_thread.join(timeout=5)
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
