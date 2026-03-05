# utils/logger.py
#
# Centralised logger — use this instead of print() in all modules.
# Writes to both the console and outputs/logs/run.log
#
# Usage:
#   from utils.logger import get_logger
#   log = get_logger(__name__)
#   log.info("Training started")
#   log.warning("No checkpoint found")
#   log.error("File not found")

import logging
import os
from datetime import datetime


LOG_DIR = "outputs/logs"
LOG_FILE = os.path.join(LOG_DIR, "run.log")


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger that writes to console + log file.

    Parameters
    ----------
    name : str — typically pass __name__ from the calling module
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger
