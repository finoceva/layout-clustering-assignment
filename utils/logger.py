"""
Custom Logger Configuration

Provides standardized logging setup for the layout clustering project.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger


def setup_logger(level: str = "INFO", log_file: Optional[Path] = None, format_str: Optional[str] = None) -> None:
    """
    Setup custom logger configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging output
        format_str: Custom format string
    """
    # Remove default logger
    logger.remove()

    # Default format
    if format_str is None:
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        )

    # Console output
    logger.add(sys.stdout, format=format_str, level=level, colorize=True)

    # File output if specified
    if log_file:
        logger.add(
            log_file,
            format=format_str,
            level=level,
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="7 days",  # Keep logs for 7 days
        )


def get_logger(name: str) -> "Logger":
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


# Convenience function for quick setup
def setup_project_logger(log_dir: Optional[Path] = None) -> None:
    """
    Setup logger for the layout clustering project.

    Args:
        log_dir: Directory to store log files
    """
    log_file = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "layout_clustering.log"

    setup_logger(level="INFO", log_file=log_file)
