"""Signal outcome evaluator.

Runs every hour. Evaluates pending buy signals:
  1. Sell signal match — exit price = close of first warm candle after the sell signal.
  2. Timeout — if no sell signal after 20 days, exit price = latest warm candle close.

Also tracks 24h metrics (gain_24h_pct, max_gain_24h_pct) while warm candles
still cover the 24h window after signal emission (i.e. when signal is 24-48h old).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.agent import storage
from src.agent.db import open_db
from src.agent.models import AppConfig

logger = logging.getLogger(__name__)

_TIMEOUT = timedelta(days=20)
_24H = timedelta(hours=24)
_24H_GRACE = timedelta(hours=48)


def evaluate_pending_signals(config: AppConfig, reference_time: datetime | None = None) -> None:
    now = reference_time or datetime.now(timezone.utc)

    with open_db(config.data_dir) as con:
        pending = con.execute(
            """SELECT signal_id, pair, emitted_at, price_at_signal, gain_24h_pct
               FROM signals WHERE direction='buy' AND gain_pct IS NULL"""
        ).fetchall()

        if not pending:
            return

        sell_rows = con.execute(
            "SELECT pair, emitted_at FROM signals WHERE direction='sell'"
        ).fetchall()

    sell_index: dict[str, list[datetime]] = {}
    for r in sell_rows:
        dt = _parse_dt(r["emitted_at"])
        if dt is not None:
            sell_index.setdefault(r["pair"], []).append(dt)

    for row in pending:
        signal_id = row["signal_id"]
        pair = row["pair"]
        emitted_at = _parse_dt(row["emitted_at"])
        price_at_signal = row["price_at_signal"]

        if emitted_at is None or price_at_signal is None:
            continue

        if row["gain_24h_pct"] is None:
            age = now - emitted_at
            if _24H <= age <= _24H_GRACE:
                metrics = _compute_24h_metrics(pair, emitted_at, price_at_signal, config)
                if metrics is not None:
                    with open_db(config.data_dir) as con:
                        con.execute(
                            "UPDATE signals SET gain_24h_pct=?, max_gain_24h_pct=? WHERE signal_id=?",
                            (metrics["gain_24h_pct"], metrics["max_gain_24h_pct"], signal_id),
                        )

        outcome = _resolve_outcome(pair, emitted_at, price_at_signal, now, sell_index, config)
        if outcome is not None:
            with open_db(config.data_dir) as con:
                con.execute(
                    """UPDATE signals SET evaluated_at=?, exit_price=?, exit_reason=?, gain_pct=?
                       WHERE signal_id=?""",
                    (outcome["evaluated_at"], outcome["exit_price"],
                     outcome["exit_reason"], outcome["gain_pct"], signal_id),
                )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_outcome(
    pair: str,
    emitted_at: datetime,
    price_at_signal: float,
    now: datetime,
    sell_index: dict[str, list[datetime]],
    config: AppConfig,
) -> dict | None:
    sells_after = [dt for dt in sell_index.get(pair, []) if dt > emitted_at]
    if sells_after:
        sell_time = min(sells_after)
        exit_price = _price_after(pair, sell_time, config)
        if exit_price is None:
            logger.warning(
                "No warm candle after sell signal for %s at %s — leaving pending", pair, sell_time
            )
            return None
        return {
            "evaluated_at": now.isoformat(),
            "exit_price": exit_price,
            "exit_reason": "sell_signal",
            "gain_pct": exit_price / price_at_signal - 1,
        }

    if now - emitted_at >= _TIMEOUT:
        exit_price = _latest_price(pair, config)
        if exit_price is None:
            logger.warning("No price data for %s — cannot evaluate timed-out signal", pair)
            return None
        return {
            "evaluated_at": now.isoformat(),
            "exit_price": exit_price,
            "exit_reason": "timeout",
            "gain_pct": exit_price / price_at_signal - 1,
        }

    return None


def _price_after(pair: str, after: datetime, config: AppConfig) -> float | None:
    """Close of the first warm candle strictly after `after`."""
    warm = storage.read_warm_candles(pair, config)
    candidates = [c for c in warm if c.hour > after]
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.hour).close


def _latest_price(pair: str, config: AppConfig) -> float | None:
    warm = storage.read_warm_candles(pair, config)
    return warm[-1].close if warm else None


def _compute_24h_metrics(
    pair: str, emitted_at: datetime, price_at_signal: float, config: AppConfig
) -> dict | None:
    warm = storage.read_warm_candles(pair, config)
    window_end = emitted_at + _24H
    candles = [c for c in warm if emitted_at <= c.hour <= window_end]
    if not candles:
        return None
    return {
        "gain_24h_pct": candles[-1].close / price_at_signal - 1,
        "max_gain_24h_pct": max(c.high for c in candles) / price_at_signal - 1,
    }


def _parse_dt(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None
