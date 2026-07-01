"""Hourly processing — downsample hot tier to warm, recompute cold, evaluate signals.

Invoke from cron hourly. Runs once and exits.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.agent import evaluator, storage
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


def _discover_pairs(config: AppConfig) -> list[str]:
    if config.pairs:
        return list(config.pairs)
    data_root = Path(config.data_dir)
    if not data_root.exists():
        return []
    return sorted(
        p.name for p in data_root.iterdir()
        if p.is_dir() and (p / "hot.ndjson").exists()
    )


def run(config: AppConfig) -> None:
    pairs = _discover_pairs(config)
    logger.info("Processing %d pairs", len(pairs))
    for pair in pairs:
        try:
            storage.downsample_hot_to_warm(pair, config)
        except Exception:
            logger.exception("Warm downsample failed for %s", pair)
        try:
            storage.recompute_cold_tier(pair, config)
        except Exception:
            logger.exception("Cold recompute failed for %s", pair)
    try:
        evaluator.evaluate_pending_signals(config)
    except Exception:
        logger.exception("Signal evaluation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto hourly processing")
    parser.add_argument("config", nargs="?", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    run(load_config(args.config))
