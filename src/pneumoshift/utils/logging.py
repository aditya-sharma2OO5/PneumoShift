"""Logging setup for research experiments."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "pneumoshift", log_file: Optional[Path | str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standard logger with console and optional file handler.

    Args:
        name: Name of the logger.
        log_file: Optional file path to write log messages.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

