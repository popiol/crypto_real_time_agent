"""Entry point for the crypto real-time agent.

Designed to be invoked by cron (or any process supervisor). On each
invocation, a PID file is checked to ensure only one instance runs at a time:
  - If the PID file exists and the recorded process is still alive, exit 0.
  - If the PID file is stale (process gone), clean it up and start normally.
  - On exit (clean or crash), remove the PID file.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import psutil

import yaml

from src.agent.loop import run
from src.agent.models import AppConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> AppConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppConfig.model_validate(raw)


def _is_running(pid: int) -> bool:
    return psutil.pid_exists(pid)


def _acquire_pid_lock(pid_file: Path) -> bool:
    """Write the current PID to pid_file and return True.

    Returns False (without writing) if another live instance is detected.
    Removes a stale PID file if the recorded process is no longer running.
    """
    if pid_file.exists():
        try:
            recorded_pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            recorded_pid = None

        if recorded_pid is not None and _is_running(recorded_pid):
            logger.info("Agent already running (PID %d) — exiting", recorded_pid)
            return False

        logger.warning(
            "Removing stale PID file (PID %s no longer running)",
            recorded_pid,
        )
        pid_file.unlink(missing_ok=True)

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _release_pid_lock(pid_file: Path) -> None:
    pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    pid_file = Path(config.data_dir) / "agent.pid"

    if not _acquire_pid_lock(pid_file):
        sys.exit(0)

    try:
        run(config)
    finally:
        _release_pid_lock(pid_file)
