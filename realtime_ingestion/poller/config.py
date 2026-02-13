"""
Configuration for GTFS/Alerts Pollers

Simple configuration - pollers only need to know:
- Where to fetch data (MTA feeds)
- Where to publish data (Pub/Sub topics)
- How often to poll
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for the poller application."""
    
    # GCP Configuration
    project_id: str
    gtfs_ace_topic: str
    gtfs_bdfm_topic: str
    alerts_topic: str
    
    # MTA Feed URLs (public, no API key required)
    gtfs_ace_url: str = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"
    gtfs_bdfm_url: str = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-bdfm"
    alerts_url: str = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/camsys%2Fsubway-alerts.json"
    
    # Polling Configuration
    poll_interval_seconds: int = 30
    request_timeout_seconds: int = 10
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    
    @classmethod
    def from_environment(cls) -> "Config":
        """Create configuration from environment variables."""
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
        
        # Build full topic paths
        gtfs_ace_topic = os.environ.get(
            "GTFS_ACE_TOPIC",
            f"projects/{project_id}/topics/gtfs-rt-ace"
        )
        gtfs_bdfm_topic = os.environ.get(
            "GTFS_BDFM_TOPIC",
            f"projects/{project_id}/topics/gtfs-rt-bdfm"
        )
        alerts_topic = os.environ.get(
            "ALERTS_TOPIC",
            f"projects/{project_id}/topics/service-alerts"
        )
        
        return cls(
            project_id=project_id,
            gtfs_ace_topic=gtfs_ace_topic,
            gtfs_bdfm_topic=gtfs_bdfm_topic,
            alerts_topic=alerts_topic,
            poll_interval_seconds=int(os.environ.get("POLL_INTERVAL", "30")),
            request_timeout_seconds=int(os.environ.get("REQUEST_TIMEOUT", "10")),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
        )


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration."""
    global _config
    if _config is None:
        _config = Config.from_environment()
    return _config
