"""
ACE-only poller entry point for integration testing.

Runs only the A/C/E feed poller — no BDFM, no alerts.
Single topic: gtfs-rt-ace.
"""

import logging
import os
import signal
import time

from gtfs_poller import GTFSPoller
from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

shutdown_requested = False


def _handle_signal(signum, frame):
    global shutdown_requested
    logger.info("Shutdown signal received.")
    shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def main():
    config = get_config()
    poller = GTFSPoller(
        feed_url=config.gtfs_ace_url,
        topic_path=config.gtfs_ace_topic,
        feed_id="ace",
        config=config,
    )

    logger.info("ACE poller started — topic: %s", config.gtfs_ace_topic)

    while not shutdown_requested:
        start = time.time()
        try:
            result = poller.poll_once()
            if result["success"]:
                logger.info(
                    "Published %d entities (%d bytes)",
                    result.get("entities", 0),
                    result.get("bytes", 0),
                )
            else:
                logger.warning("Poll failed")
        except Exception as e:
            logger.error("Error: %s", e)

        elapsed = time.time() - start
        remaining = max(0, config.poll_interval_seconds - elapsed)
        while remaining > 0 and not shutdown_requested:
            time.sleep(min(1, remaining))
            remaining -= 1

    logger.info("ACE poller stopped.")


if __name__ == "__main__":
    main()
