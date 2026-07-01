"""Daily analysis — runs the strategy updater pipeline.

Invoke from cron daily. Runs once and exits.
"""

from __future__ import annotations

import argparse
import logging

import yaml

from src.agent.models import AppConfig
from src.updater import pipeline as updater_pipeline

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


def run(config: AppConfig) -> None:
    try:
        updater_pipeline.run(config)
    except Exception:
        logger.exception("Strategy updater pipeline failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto daily analysis")
    parser.add_argument("config", nargs="?", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    run(load_config(args.config))
