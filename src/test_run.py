"""Test runner — replays historical snapshots through the full pipeline in a single loop.

Simulates pull + process in sequence for each snapshot, then exits.
The analyze step (LLM updater) is excluded; run src/analyze.py separately if needed.
"""

from __future__ import annotations

import argparse
import logging

import yaml

from src.agent import backtest_collector, portfolio as _portfolio, storage
from src.agent.loop import persist_signals, run_strategy
from src.agent.models import AppConfig
from src.analyze import run as analyze_run
from src.process import run as process_run

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


_ANALYZE_INTERVAL_CYCLES = 24  # one snapshot per hour → analyze once per day


def run(config: AppConfig, clear: bool = False) -> None:
    if clear:
        logger.info("Clearing all data for test run")
        storage.reset_for_backtest(config)
    logger.info("Starting test run from %s", config.backtest_data_dir)

    cycle = 0
    while True:
        ticks = backtest_collector.next_snapshot(config)
        if ticks is None:
            logger.info("Backtest data exhausted — done")
            break
        if not ticks:
            continue

        logger.info("Cycle %d: %s (%d pairs)", cycle + 1, ticks[0].polled_at.isoformat(), len(ticks))

        try:
            storage.write_ticks(ticks, config, reference_time=ticks[0].polled_at)
        except Exception:
            logger.exception("Storage write failed")

        signals = run_strategy(ticks, config)
        persist_signals(signals, config)
        _portfolio.run_cycle(ticks, signals, config)
        process_run(config, reference_time=ticks[0].polled_at)

        cycle += 1
        if cycle % _ANALYZE_INTERVAL_CYCLES == 0:
            analyze_run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto test runner (historical replay)")
    parser.add_argument("config", nargs="?", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--clear", action="store_true", help="Delete the data directory before starting")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run(load_config(args.config), clear=args.clear)
