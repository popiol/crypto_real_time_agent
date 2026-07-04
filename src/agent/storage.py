"""Tiered storage for hot, warm, and cold data per currency pair."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.agent.db import open_db
from src.agent.models import AppConfig, ColdMonth, Tick, WarmCandle

logger = logging.getLogger(__name__)


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _tick_from_row(row) -> Tick:
    return Tick(
        pair=row["pair"],
        polled_at=_parse_dt(row["polled_at"]),
        last_price=row["last_price"],
        bid_price=row["bid_price"],
        bid_volume=row["bid_volume"],
        ask_price=row["ask_price"],
        ask_volume=row["ask_volume"],
        volume_24h=row["volume_24h"],
        mid_price=row["mid_price"],
        spread_abs=row["spread_abs"],
        spread_rel=row["spread_rel"],
        order_book=json.loads(row["order_book"]) if row["order_book"] else None,
    )


def _candle_from_row(row) -> WarmCandle:
    return WarmCandle(
        hour=_parse_dt(row["hour"]),
        open_price=row["open_price"],
        high=row["high"],
        low=row["low"],
        close=row["close"],
        avg_spread_rel=row["avg_spread_rel"],
    )


def _cold_from_row(row) -> ColdMonth:
    return ColdMonth(
        month=row["month"],
        min_price=row["min_price"],
        max_price=row["max_price"],
        avg_price=row["avg_price"],
        avg_daily_spread=row["avg_daily_spread"],
        candle_count=row["candle_count"],
        last_candle_hour=_parse_dt(row["last_candle_hour"]),
    )


# ── Hot tier ──────────────────────────────────────────────────────────────────


def read_ticks(pair: str, config: AppConfig) -> list[Tick]:
    with open_db(config.data_dir) as con:
        rows = con.execute(
            "SELECT * FROM hot_ticks WHERE pair=? ORDER BY polled_at ASC", (pair,)
        ).fetchall()
    return [_tick_from_row(r) for r in rows]


def write_ticks(
    ticks: list[Tick],
    config: AppConfig,
    reference_time: datetime | None = None,
) -> None:
    now = reference_time or datetime.now(timezone.utc)
    cutoff_str = (now - timedelta(seconds=config.hot_tier_retention_seconds)).isoformat()

    with open_db(config.data_dir) as con:
        con.executemany(
            """INSERT INTO hot_ticks
               (pair, polled_at, last_price, bid_price, bid_volume,
                ask_price, ask_volume, volume_24h, mid_price, spread_abs, spread_rel, order_book)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    t.pair, t.polled_at.isoformat(),
                    t.last_price, t.bid_price, t.bid_volume,
                    t.ask_price, t.ask_volume, t.volume_24h,
                    t.mid_price, t.spread_abs, t.spread_rel,
                    t.order_book.model_dump_json() if t.order_book else None,
                )
                for t in ticks
            ],
        )
        for pair in {t.pair for t in ticks}:
            _prune_pair(pair, cutoff_str, con)


def _prune_pair(pair: str, cutoff_str: str, con) -> None:
    expired = con.execute(
        "SELECT * FROM hot_ticks WHERE pair=? AND polled_at<? ORDER BY polled_at ASC",
        (pair, cutoff_str),
    ).fetchall()

    if expired:
        existing_rows = con.execute(
            """SELECT * FROM (
                 SELECT * FROM warm_candles WHERE pair=? ORDER BY hour DESC LIMIT 24
               ) ORDER BY hour ASC""",
            (pair,),
        ).fetchall()
        merged = _merge_into_candles(
            [_candle_from_row(r) for r in existing_rows],
            [_tick_from_row(r) for r in expired],
        )
        _upsert_warm_candles(merged, pair, con)

    con.execute(
        "DELETE FROM hot_ticks WHERE pair=? AND polled_at<?", (pair, cutoff_str)
    )


def _upsert_warm_candles(candles: list[WarmCandle], pair: str, con) -> None:
    con.executemany(
        """INSERT INTO warm_candles (pair, hour, open_price, high, low, close, avg_spread_rel)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(pair, hour) DO UPDATE SET
             open_price=excluded.open_price, high=excluded.high,
             low=excluded.low, close=excluded.close,
             avg_spread_rel=excluded.avg_spread_rel""",
        [
            (pair, c.hour.isoformat(), c.open_price, c.high, c.low, c.close, c.avg_spread_rel)
            for c in candles
        ],
    )
    con.execute(
        """DELETE FROM warm_candles WHERE pair=? AND hour NOT IN (
             SELECT hour FROM warm_candles WHERE pair=? ORDER BY hour DESC LIMIT 24
           )""",
        (pair, pair),
    )


# ── Warm tier ─────────────────────────────────────────────────────────────────


def read_warm_candles(pair: str, config: AppConfig) -> list[WarmCandle]:
    with open_db(config.data_dir) as con:
        rows = con.execute(
            """SELECT * FROM (
                 SELECT * FROM warm_candles WHERE pair=? ORDER BY hour DESC LIMIT 24
               ) ORDER BY hour ASC""",
            (pair,),
        ).fetchall()
    return [_candle_from_row(r) for r in rows]


def write_warm_candles(candles: list[WarmCandle], pair: str, config: AppConfig) -> None:
    with open_db(config.data_dir) as con:
        _upsert_warm_candles(candles, pair, con)


def _merge_into_candles(
    existing: list[WarmCandle], new_ticks: list[Tick]
) -> list[WarmCandle]:
    """Merge ticks into existing hourly candles and return the last 24."""
    candle_map: dict[datetime, WarmCandle] = {c.hour: c for c in existing}

    by_hour: dict[datetime, list[Tick]] = {}
    for tick in new_ticks:
        hour = tick.polled_at.replace(minute=0, second=0, microsecond=0)
        by_hour.setdefault(hour, []).append(tick)

    for hour, ticks in by_hour.items():
        ticks.sort(key=lambda t: t.polled_at)
        prices = [t.last_price for t in ticks]
        avg_spread = sum(t.spread_rel for t in ticks) / len(ticks)
        if hour in candle_map:
            c = candle_map[hour]
            candle_map[hour] = WarmCandle(
                hour=hour,
                open_price=c.open_price,
                high=max(c.high, max(prices)),
                low=min(c.low, min(prices)),
                close=prices[-1],
                avg_spread_rel=avg_spread,
            )
        else:
            candle_map[hour] = WarmCandle(
                hour=hour,
                open_price=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                avg_spread_rel=avg_spread,
            )

    return sorted(candle_map.values(), key=lambda c: c.hour)[-24:]


def downsample_hot_to_warm(pair: str, config: AppConfig) -> None:
    ticks = read_ticks(pair, config)
    if not ticks:
        return
    existing = read_warm_candles(pair, config)
    write_warm_candles(_merge_into_candles(existing, ticks), pair, config)


# ── Cold tier ─────────────────────────────────────────────────────────────────


def read_cold_months(pair: str, config: AppConfig) -> list[ColdMonth]:
    with open_db(config.data_dir) as con:
        rows = con.execute(
            "SELECT * FROM cold_months WHERE pair=? ORDER BY month ASC", (pair,)
        ).fetchall()
    return [_cold_from_row(r) for r in rows]


def write_cold_months(months: list[ColdMonth], pair: str, config: AppConfig) -> None:
    with open_db(config.data_dir) as con:
        con.executemany(
            """INSERT INTO cold_months
               (pair, month, min_price, max_price, avg_price,
                avg_daily_spread, candle_count, last_candle_hour)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(pair, month) DO UPDATE SET
                 min_price=excluded.min_price, max_price=excluded.max_price,
                 avg_price=excluded.avg_price, avg_daily_spread=excluded.avg_daily_spread,
                 candle_count=excluded.candle_count, last_candle_hour=excluded.last_candle_hour""",
            [
                (pair, m.month, m.min_price, m.max_price, m.avg_price,
                 m.avg_daily_spread, m.candle_count, m.last_candle_hour.isoformat())
                for m in months
            ],
        )


def recompute_cold_tier(pair: str, config: AppConfig) -> None:
    warm = read_warm_candles(pair, config)
    if not warm:
        return

    month_map: dict[str, ColdMonth] = {m.month: m for m in read_cold_months(pair, config)}

    by_month: dict[str, list[WarmCandle]] = {}
    for candle in warm:
        key = candle.hour.strftime("%Y-%m")
        by_month.setdefault(key, []).append(candle)

    for month_key, candles in by_month.items():
        candles.sort(key=lambda c: c.hour)
        if month_key in month_map:
            old = month_map[month_key]
            new = [c for c in candles if c.hour > old.last_candle_hour]
            if not new:
                continue
            prices = [c.close for c in new]
            spreads = [c.avg_spread_rel for c in new]
            total = old.candle_count + len(new)
            month_map[month_key] = ColdMonth(
                month=month_key,
                min_price=min(old.min_price, min(prices)),
                max_price=max(old.max_price, max(prices)),
                avg_price=(old.avg_price * old.candle_count + sum(prices)) / total,
                avg_daily_spread=(old.avg_daily_spread * old.candle_count + sum(spreads)) / total,
                candle_count=total,
                last_candle_hour=new[-1].hour,
            )
        else:
            prices = [c.close for c in candles]
            spreads = [c.avg_spread_rel for c in candles]
            n = len(candles)
            month_map[month_key] = ColdMonth(
                month=month_key,
                min_price=min(prices),
                max_price=max(prices),
                avg_price=sum(prices) / n,
                avg_daily_spread=sum(spreads) / n,
                candle_count=n,
                last_candle_hour=candles[-1].hour,
            )

    write_cold_months(sorted(month_map.values(), key=lambda m: m.month), pair, config)


# ── Backtest reset ────────────────────────────────────────────────────────────


def reset_for_backtest(config: AppConfig) -> None:
    """Wipe all runtime data so a test run starts from a clean slate."""
    with open_db(config.data_dir) as con:
        for table in ("hot_ticks", "warm_candles", "cold_months", "signals"):
            con.execute(f"DELETE FROM {table}")

    for subdir in ("rules", "state"):
        target = Path(config.data_dir) / subdir
        if target.exists():
            shutil.rmtree(target)


# ── Signal ledger ─────────────────────────────────────────────────────────────


def read_signals(config: AppConfig) -> list[dict]:
    """Return all signal records with outcome nested as a dict (or None if unresolved)."""
    with open_db(config.data_dir) as con:
        rows = con.execute(
            "SELECT * FROM signals ORDER BY emitted_at ASC"
        ).fetchall()
    return [_signal_row_to_dict(r) for r in rows]


def _signal_row_to_dict(row) -> dict:
    outcome = None
    if row["gain_pct"] is not None:
        outcome = {
            "evaluated_at": row["evaluated_at"],
            "exit_price": row["exit_price"],
            "exit_reason": row["exit_reason"],
            "gain_pct": row["gain_pct"],
        }
        if row["gain_24h_pct"] is not None:
            outcome["gain_24h_pct"] = row["gain_24h_pct"]
            outcome["max_gain_24h_pct"] = row["max_gain_24h_pct"]
    return {
        "signal_id": row["signal_id"],
        "direction": row["direction"],
        "pair": row["pair"],
        "rule_id": row["rule_id"],
        "emitted_at": row["emitted_at"],
        "price_at_signal": row["price_at_signal"],
        "confidence": row["confidence"],
        "outcome": outcome,
    }
