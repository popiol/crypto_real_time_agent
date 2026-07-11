"""Signal outcome evaluator.

Runs every hour. Evaluates pending buy signals:
  1. Sell signal match — exit price = sell signal price (treated as fill price).
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
            """SELECT signal_id, pair, emitted_at, price_at_signal
               FROM signals WHERE direction='buy' AND gain_pct IS NULL"""
        ).fetchall()
        needs_24h = con.execute(
            """SELECT signal_id, pair, emitted_at, price_at_signal
               FROM signals WHERE direction='buy' AND gain_24h_pct IS NULL"""
        ).fetchall()
        sell_rows = con.execute(
            "SELECT pair, emitted_at, price_at_signal FROM signals WHERE direction='sell'"
        ).fetchall()

    sell_index = _build_sell_index(sell_rows)
    _update_24h_metrics(needs_24h, now, config)
    _resolve_pending(pending, now, sell_index, config)


def _build_sell_index(sell_rows) -> dict[str, list[tuple[datetime, float]]]:
    index: dict[str, list[tuple[datetime, float]]] = {}
    for r in sell_rows:
        dt = _parse_dt(r["emitted_at"])
        price = r["price_at_signal"]
        if dt is not None and price is not None:
            index.setdefault(r["pair"], []).append((dt, price))
    return index


def _update_24h_metrics(rows, now: datetime, config: AppConfig) -> None:
    for row in rows:
        emitted_at = _parse_dt(row["emitted_at"])
        price_at_signal = row["price_at_signal"]
        if emitted_at is None or price_at_signal is None:
            continue
        age = now - emitted_at
        if not (_24H <= age <= _24H_GRACE):
            continue
        metrics = _compute_24h_metrics(row["pair"], emitted_at, price_at_signal, config)
        if metrics is not None:
            with open_db(config.data_dir) as con:
                con.execute(
                    "UPDATE signals SET gain_24h_pct=?, max_gain_24h_pct=? WHERE signal_id=?",
                    (metrics["gain_24h_pct"], metrics["max_gain_24h_pct"], row["signal_id"]),
                )


def _resolve_pending(
    pending, now: datetime, sell_index: dict[str, list[tuple[datetime, float]]], config: AppConfig
) -> None:
    for row in pending:
        emitted_at = _parse_dt(row["emitted_at"])
        price_at_signal = row["price_at_signal"]
        if emitted_at is None or price_at_signal is None:
            continue
        outcome = _resolve_outcome(row["pair"], emitted_at, price_at_signal, now, sell_index, config)
        if outcome is not None:
            with open_db(config.data_dir) as con:
                con.execute(
                    """UPDATE signals SET evaluated_at=?, exit_price=?, exit_reason=?, gain_pct=?
                       WHERE signal_id=?""",
                    (outcome["evaluated_at"], outcome["exit_price"],
                     outcome["exit_reason"], outcome["gain_pct"], row["signal_id"]),
                )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_outcome(
    pair: str,
    emitted_at: datetime,
    price_at_signal: float,
    now: datetime,
    sell_index: dict[str, list[tuple[datetime, float]]],
    config: AppConfig,
) -> dict | None:
    sells_after = [(dt, p) for dt, p in sell_index.get(pair, []) if dt > emitted_at]
    if sells_after:
        _, sell_price = min(sells_after, key=lambda x: x[0])
        return {
            "evaluated_at": now.isoformat(),
            "exit_price": sell_price,
            "exit_reason": "sell_signal",
            "gain_pct": sell_price / price_at_signal - 1,
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
