"""Shared logging setup for all entry-point scripts.

Configures the root logger once and quiets third-party libraries that log
noisy INFO-level request/wiring lines (HTTP request traces, SDK setup
messages) so application logs aren't drowned out.
"""

from __future__ import annotations

import logging

_NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "google_genai",
    "google.genai",
    "google.auth",
    "urllib3",
]


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
