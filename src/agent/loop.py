"""Main polling loop.

Runs continuously:
  1. Collect ticks from Kraken (Ticker + Depth).
  2. Persist ticks to the hot tier.
  3. Run find_buy_signals() from the strategy module.
  4. Persist any buy signals to the signal ledger.
  5. Wait until min_poll_interval has elapsed since the cycle started, then repeat.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from src.agent import collector, storage
from src.agent.models import AppConfig, BuySignal, PairData
from src.strategy.strategy import find_buy_signals

logger = logging.getLogger(__name__)


def _append_signals(signals: list[BuySignal], ledger_path: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fh:
        for signal in signals:
            record = {
                "signal_id": str(uuid.uuid4()),
                "pair": signal.pair,
                "rule_id": signal.rule_id,
                "emitted_at": signal.timestamp.isoformat(),
                "price_at_signal": signal.price,
                "confidence": signal.confidence,
                "outcome": None,
            }
            fh.write(json.dumps(record) + "\n")


_WARM_REFRESH_INTERVAL_S = 3600.0


def _refresh_warm_tier(pairs: set[str], config: AppConfig) -> None:
    logger.info("Refreshing warm tier for %d pairs", len(pairs))
    for pair in pairs:
        try:
            candles = collector.fetch_warm_candles(pair, config)
            storage.write_warm_candles(candles, pair, config)
        except Exception:
            logger.exception("Warm refresh failed for %s", pair)


def run(config: AppConfig) -> None:
    """Start the polling loop. Runs until interrupted."""
    ledger_path = Path(config.data_dir) / "signals.ndjson"
    logger.info("Starting polling loop. Pairs: %s", config.pairs or "auto-discover all USD pairs")

    last_warm_refresh = 0.0

    while True:
        cycle_start = time.monotonic()

        # --- 1. Collect ---
        try:
            ticks = collector.collect(config)
        except Exception:
            logger.exception("Collection failed")
            ticks = []

        # --- 2. Persist to hot tier ---
        if ticks:
            try:
                storage.write_ticks(ticks, config)
            except Exception:
                logger.exception("Storage write failed")

        # --- 2b. Refresh warm tier (hourly) ---
        if ticks and cycle_start - last_warm_refresh >= _WARM_REFRESH_INTERVAL_S:
            _refresh_warm_tier({t.pair for t in ticks}, config)
            last_warm_refresh = cycle_start

        # --- 3. Run strategy ---
        try:
            market_data = {
                tick.pair: PairData(
                    hot=storage.read_ticks(tick.pair, config),
                    warm=storage.read_warm_candles(tick.pair, config),
                )
                for tick in ticks
            }
            signals: list[BuySignal] = find_buy_signals(market_data)
        except Exception:
            logger.exception("Strategy execution failed")
            signals = []

        # --- 4. Persist signals ---
        if signals:
            logger.info("Buy signals: %s", [(s.rule_id, s.pair) for s in signals])
            try:
                _append_signals(signals, ledger_path)
            except Exception:
                logger.exception("Failed to write signals")

        # --- 5. Sleep for the remainder of min_poll_interval ---
        elapsed = time.monotonic() - cycle_start
        sleep_for = max(0.0, config.min_poll_interval_seconds - elapsed)
        if sleep_for:
            time.sleep(sleep_for)
