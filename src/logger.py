"""
logger.py — Structured logging using Loguru with Rich console output.
"""

import sys
import os
from loguru import logger
from src.config import settings


def setup_logger() -> None:
    """Configure Loguru logger with console + file sinks."""

    # Remove default handler
    logger.remove()

    # Console handler — coloured, human-readable
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — JSON-structured for log aggregation
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
    logger.add(
        settings.log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} — {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        serialize=False,
    )

    logger.info("Logger initialised. Level={}", settings.log_level)


# Initialise on import
setup_logger()

__all__ = ["logger"]
