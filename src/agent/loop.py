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

from src.agent import backtest_collector, collector, evaluator, storage
from src.agent.models import AppConfig, BuySignal, PairData, SellSignal, Tick
from src.strategy.strategy import find_signals
from src.updater import pipeline as updater_pipeline

logger = logging.getLogger(__name__)


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


_WARM_REFRESH_INTERVAL_S = 3600.0
_UPDATER_INTERVAL_S = 86400.0


def _refresh_tiers(pairs: set[str], config: AppConfig) -> None:
    logger.info("Refreshing warm and cold tiers for %d pairs", len(pairs))
    for pair in pairs:
        try:
            storage.downsample_hot_to_warm(pair, config)
        except Exception:
            logger.exception("Warm refresh failed for %s", pair)
        try:
            storage.recompute_cold_tier(pair, config)
        except Exception:
            logger.exception("Cold recompute failed for %s", pair)
    try:
        evaluator.evaluate_pending_signals(config)
    except Exception:
        logger.exception("Signal evaluation failed")


def _collect_ticks(config: AppConfig) -> list[Tick] | None:
    """Return ticks for this cycle, or None to signal exhaustion (test mode only)."""
    if config.test_mode:
        return backtest_collector.next_snapshot(config)
    try:
        return collector.collect(config)
    except Exception:
        logger.exception("Collection failed")
        return []


def _persist_ticks(ticks: list[Tick], config: AppConfig) -> None:
    if not ticks:
        return
    reference_time = ticks[0].polled_at if config.test_mode else None
    try:
        storage.write_ticks(ticks, config, reference_time=reference_time)
    except Exception:
        logger.exception("Storage write failed")


def _maybe_refresh_tiers(
    ticks: list[Tick], cycle_start: float, last_refresh: float, config: AppConfig
) -> float:
    should_refresh = ticks and (
        config.test_mode or cycle_start - last_refresh >= _WARM_REFRESH_INTERVAL_S
    )
    if should_refresh:
        _refresh_tiers({t.pair for t in ticks}, config)
        return cycle_start
    return last_refresh


def _maybe_run_updater(cycle_start: float, last_run: float, config: AppConfig) -> float:
    if config.test_mode:
        return last_run
    if cycle_start - last_run >= _UPDATER_INTERVAL_S:
        try:
            updater_pipeline.run(config)
        except Exception:
            logger.exception("Strategy Updater pipeline failed")
        return cycle_start
    return last_run


_MIN_VOLUME_24H_USD = 1_000.0


def _run_strategy(ticks: list, config: AppConfig) -> list[BuySignal | SellSignal]:
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


def _persist_signals(signals: list[BuySignal | SellSignal], ledger_path: Path) -> None:
    if not signals:
        return
    logger.info("Signals: %s", [(s.rule_id, s.pair, type(s).__name__) for s in signals])
    try:
        _append_signals(signals, ledger_path)
    except Exception:
        logger.exception("Failed to write signals")


def run(config: AppConfig) -> None:
    """Start the polling loop. Runs until interrupted."""
    ledger_path = Path(config.data_dir) / "signals.ndjson"
    logger.info("Starting polling loop. Pairs: %s", config.pairs or "auto-discover all USD pairs")

    last_warm_refresh = 0.0
    last_updater_run = 0.0

    while True:
        cycle_start = time.monotonic()

        ticks = _collect_ticks(config)
        if ticks is None:
            logger.info("Backtest data exhausted — stopping")
            break

        _persist_ticks(ticks, config)
        last_warm_refresh = _maybe_refresh_tiers(ticks, cycle_start, last_warm_refresh, config)
        last_updater_run = _maybe_run_updater(cycle_start, last_updater_run, config)
        _persist_signals(_run_strategy(ticks, config), ledger_path)

        if not config.test_mode:
            elapsed = time.monotonic() - cycle_start
            sleep_for = max(0.0, config.min_poll_interval_seconds - elapsed)
            if sleep_for:
                time.sleep(sleep_for)
