"""
logger.py
─────────
Lightweight logging helper for MindSaathi backend.
Drop-in replacement: swap print() for a proper logging framework later.
"""

import logging
import sys
from datetime import datetime, timezone

# ── Configure root logger ─────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_logger = logging.getLogger("neuroaid")


def log_info(message: str) -> None:
    _logger.info(message)


def log_warning(message: str) -> None:
    _logger.warning(message)


def log_error(message: str) -> None:
    _logger.error(message)


def log_debug(message: str) -> None:
    _logger.debug(message)


def log_request(endpoint: str, payload: dict) -> None:
    """Pretty-print an incoming request for debugging."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _logger.info(f"[{ts}] REQUEST → {endpoint} | payload keys: {list(payload.keys())}")


