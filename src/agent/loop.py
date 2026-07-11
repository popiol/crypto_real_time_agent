"""Real-time polling loop.

Runs continuously:
  1. Collect ticks from Kraken.
  2. Persist ticks to the hot tier.
  3. Run strategy rules and emit signals to the signal ledger.
  4. Sleep until min_poll_interval has elapsed, then repeat.
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
import uuid

from src.agent import collector, portfolio as _portfolio, storage
from src.agent.db import open_db
from src.agent.models import AppConfig, BuySignal, PairData, SellSignal, Tick
import src.strategy.strategy as _strategy

logger = logging.getLogger(__name__)

_MIN_VOLUME_24H_USD = 1_000.0


def _append_signals(signals: list[BuySignal | SellSignal], config: AppConfig) -> None:
    with open_db(config.data_dir) as con:
        con.executemany(
            """INSERT INTO signals
               (signal_id, direction, pair, rule_id, emitted_at, price_at_signal, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    str(uuid.uuid4()),
                    "sell" if isinstance(s, SellSignal) else "buy",
                    s.pair, s.rule_id, s.timestamp.isoformat(), s.price, s.confidence,
                )
                for s in signals
            ],
        )


def run_strategy(ticks: list[Tick], config: AppConfig) -> list[BuySignal | SellSignal]:
    try:
        market_data = {
            tick.pair: PairData(
                hot=storage.read_ticks(tick.pair, config),
                warm=storage.read_warm_candles(tick.pair, config),
                cold=storage.read_cold_months(tick.pair, config),
            )
            for tick in ticks
        }
        importlib.reload(sys.modules["src.strategy.strategy"])
        signals = list(_strategy.find_signals(market_data))
        volume_usd = {t.pair: t.volume_24h * t.last_price for t in ticks}
        return [
            s for s in signals
            if isinstance(s, SellSignal) or volume_usd.get(s.pair, 0.0) >= _MIN_VOLUME_24H_USD
        ]
    except Exception:
        logger.exception("Strategy execution failed")
        return []


def persist_signals(signals: list[BuySignal | SellSignal], config: AppConfig) -> None:
    if not signals:
        return
    buys = sum(1 for s in signals if isinstance(s, BuySignal))
    logger.info("%d signal(s): %d buy, %d sell", len(signals), buys, len(signals) - buys)
    try:
        _append_signals(signals, config)
    except Exception:
        logger.exception("Failed to write signals")


def run(config: AppConfig) -> None:
    """Start the live polling loop. Runs until interrupted."""
    logger.info("Starting pull loop. Pairs: %s", config.pairs or "auto-discover all USD pairs")

    while True:
        cycle_start = time.monotonic()

        try:
            ticks = collector.collect(config)
        except Exception:
            logger.exception("Collection failed")
            ticks = []

        try:
            storage.write_ticks(ticks, config)
        except Exception:
            logger.exception("Storage write failed")

        signals = run_strategy(ticks, config)
        persist_signals(signals, config)
        _portfolio.run_cycle(ticks, signals, config)

        elapsed = time.monotonic() - cycle_start
        sleep_for = max(0.0, config.min_poll_interval_seconds - elapsed)
        if sleep_for:
            time.sleep(sleep_for)
