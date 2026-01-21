"""Logging utilities for comprehensive debugging."""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(component_name: str, log_level: str = "INFO") -> logging.Logger:
    """Setup structured logging for a component.
    
    Args:
        component_name: Name of the component (e.g., "preprocess", "train")
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(component_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    
    return logger


def log_component_start(logger: logging.Logger, component_name: str, **kwargs):
    """Log component start with parameters."""
    logger.info("=" * 80)
    logger.info(f"Starting {component_name} component")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    if kwargs:
        logger.info("Parameters:")
        for key, value in kwargs.items():
            logger.info(f"  {key}: {value}")
    logger.info("=" * 80)


def log_component_end(logger: logging.Logger, component_name: str, success: bool = True):
    """Log component completion."""
    status = "SUCCESS" if success else "FAILURE"
    logger.info("=" * 80)
    logger.info(f"{component_name} component completed: {status}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)
