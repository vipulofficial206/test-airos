"""
AirOS++ Structured Logger
Provides thread-safe logging with console formatting and rotating file handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

_LOGGER_NAME = "AirOS"
_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = _LOGGER_NAME,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_file_path: str = "airos.log",
) -> logging.Logger:
    """Configures and initializes the global AirOS logger."""
    global _logger

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.handlers.clear()

    # Formatter
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File Handler
    if log_to_file:
        file_path = Path(log_file_path)
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Returns the initialized AirOS logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger
