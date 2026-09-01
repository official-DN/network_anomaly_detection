"""
Centralized logging configuration using Loguru.
Creates rotating log files for application, detection, and errors.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: str = "./logs", log_level: str = "INFO") -> None:
    """Configure application logging."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # Application log
    logger.add(
        log_path / "application.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        filter=lambda record: record["level"].name not in ("ERROR",),
    )

    # Detection log
    logger.add(
        log_path / "detection.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        rotation="50 MB",
        retention="14 days",
        filter=lambda record: "detection" in record["extra"].get("module", ""),
    )

    # Error log
    logger.add(
        log_path / "error.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        rotation="5 MB",
        retention="30 days",
        backtrace=True,
        diagnose=True,
    )

    logger.info("Logging initialized")


# Initialize with defaults — override in main.py
setup_logger()
