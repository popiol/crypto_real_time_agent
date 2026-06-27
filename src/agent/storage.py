"""Hot-tier storage: append-only NDJSON files, one per currency pair.

Each line is a JSON-serialised Tick. Old ticks outside the retention window
are pruned on every write to keep files from growing unboundedly.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.agent.models import AppConfig, Tick

logger = logging.getLogger(__name__)


def _hot_path(data_dir: str, pair: str) -> Path:
    return Path(data_dir) / pair / "hot.ndjson"


def write_ticks(ticks: list[Tick], config: AppConfig) -> None:
    """Append ticks to the hot tier and prune entries outside the retention window."""
    for tick in ticks:
        path = _hot_path(config.data_dir, tick.pair)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(tick.model_dump_json() + "\n")

    # Prune old ticks for each affected pair
    affected_pairs = {t.pair for t in ticks}
    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=config.hot_tier_retention_seconds
    )
    for pair in affected_pairs:
        _prune(pair, cutoff, config.data_dir)


def read_ticks(pair: str, config: AppConfig) -> list[Tick]:
    """Read all ticks currently in the hot tier for the given pair."""
    path = _hot_path(config.data_dir, pair)
    if not path.exists():
        return []

    ticks: list[Tick] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ticks.append(Tick.model_validate_json(line))
            except Exception as exc:
                logger.warning(
                    "Skipping malformed line %d in %s: %s", lineno, path, exc
                )
    return ticks


def _prune(pair: str, cutoff: datetime, data_dir: str) -> None:
    """Rewrite the hot file keeping only ticks at or after cutoff."""
    path = _hot_path(data_dir, pair)
    if not path.exists():
        return

    kept: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                polled_at = datetime.fromisoformat(raw["polled_at"])
                if polled_at.tzinfo is None:
                    polled_at = polled_at.replace(tzinfo=timezone.utc)
                if polled_at >= cutoff:
                    kept.append(line)
            except Exception:
                # Keep malformed lines to avoid silent data loss; they will be
                # reported by read_ticks when accessed.
                kept.append(line)

    tmp_path = path.with_suffix(".ndjson.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            for line in kept:
                fh.write(line + "\n")
        os.replace(tmp_path, path)
    except Exception:
        logger.exception("Failed to prune hot tier for %s", pair)
        tmp_path.unlink(missing_ok=True)
