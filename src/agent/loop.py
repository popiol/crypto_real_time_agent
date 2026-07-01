"""Real-time polling loop.

Runs continuously:
  1. Collect ticks from Kraken.
  2. Persist ticks to the hot tier.
  3. Run strategy rules and emit signals to the signal ledger.
  4. Sleep until min_poll_interval has elapsed, then repeat.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from src.agent import collector, storage
from src.agent.models import AppConfig, BuySignal, PairData, SellSignal, Tick
from src.strategy.strategy import find_signals

logger = logging.getLogger(__name__)

_MIN_VOLUME_24H_USD = 1_000.0


def _append_signals(signals: list[BuySignal | SellSignal], ledger_path: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fh:
        for signal in signals:
            record = {
                "signal_id": str(uuid.uuid4()),
                "direction": "sell" if isinstance(signal, SellSignal) else "buy",
                "pair": signal.pair,
                "rule_id": signal.rule_id,
                "emitted_at": signal.timestamp.isoformat(),
                "price_at_signal": signal.price,
                "confidence": signal.confidence,
                "outcome": None,
            }
            fh.write(json.dumps(record) + "\n")


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
        signals = list(find_signals(market_data))
        volume_usd = {t.pair: t.volume_24h * t.last_price for t in ticks}
        return [
            s for s in signals
            if isinstance(s, SellSignal) or volume_usd.get(s.pair, 0.0) >= _MIN_VOLUME_24H_USD
        ]
    except Exception:
        logger.exception("Strategy execution failed")
        return []


def persist_signals(signals: list[BuySignal | SellSignal], ledger_path: Path) -> None:
    if not signals:
        return
    logger.info("Signals: %s", [(s.rule_id, s.pair, type(s).__name__) for s in signals])
    try:
        _append_signals(signals, ledger_path)
    except Exception:
        logger.exception("Failed to write signals")


def run(config: AppConfig) -> None:
    """Start the live polling loop. Runs until interrupted."""
    ledger_path = Path(config.data_dir) / "signals.ndjson"
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

        persist_signals(run_strategy(ticks, config), ledger_path)

        elapsed = time.monotonic() - cycle_start
        sleep_for = max(0.0, config.min_poll_interval_seconds - elapsed)
        if sleep_for:
            time.sleep(sleep_for)
