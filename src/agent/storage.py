"""Tiered storage for hot, warm, and cold data per currency pair."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.agent.models import AppConfig, ColdMonth, Tick, WarmCandle

logger = logging.getLogger(__name__)


def _hot_path(data_dir: str, pair: str) -> Path:
    return Path(data_dir) / pair / "hot.ndjson"


def _warm_path(data_dir: str, pair: str) -> Path:
    return Path(data_dir) / pair / "warm.json"


def read_warm_candles(pair: str, config: AppConfig) -> list[WarmCandle]:
    path = _warm_path(config.data_dir, pair)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    candles: list[WarmCandle] = []
    for entry in raw:
        try:
            candles.append(WarmCandle.model_validate(entry))
        except Exception as exc:
            logger.warning("Skipping malformed warm candle for %s: %s", pair, exc)
    return candles


def write_warm_candles(candles: list[WarmCandle], pair: str, config: AppConfig) -> None:
    path = _warm_path(config.data_dir, pair)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump([c.model_dump(mode="json") for c in candles], fh)
        os.replace(tmp_path, path)
    except Exception:
        logger.exception("Failed to write warm candles for %s", pair)
        tmp_path.unlink(missing_ok=True)


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
    """Merge all current hot-tier ticks into the warm tier."""
    ticks = read_ticks(pair, config)
    if not ticks:
        return
    existing = read_warm_candles(pair, config)
    write_warm_candles(_merge_into_candles(existing, ticks), pair, config)


def write_ticks(ticks: list[Tick], config: AppConfig) -> None:
    """Append ticks to the hot tier and prune entries outside the retention window."""
    for tick in ticks:
        path = _hot_path(config.data_dir, tick.pair)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(tick.model_dump_json() + "\n")

    affected_pairs = {t.pair for t in ticks}
    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=config.hot_tier_retention_seconds
    )
    for pair in affected_pairs:
        _prune_and_downsample(pair, cutoff, config)


def read_ticks(pair: str, config: AppConfig) -> list[Tick]:
    """Read all ticks currently in the hot tier for the given pair."""
    path = _hot_path(config.data_dir, pair)
    if not path.exists():
        return []

    ticks: list[Tick] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ticks.append(Tick.model_validate_json(line))
            except Exception as exc:
                logger.warning(
                    "Skipping malformed line %d in %s: %s", lineno, path, exc
                )
    return ticks


def _classify_line(
    line: str, cutoff: datetime
) -> tuple[bool, Tick | None]:
    """Return (keep, tick_or_none). keep=True → retain in hot tier."""
    try:
        raw = json.loads(line)
        polled_at = datetime.fromisoformat(raw["polled_at"])
        if polled_at.tzinfo is None:
            polled_at = polled_at.replace(tzinfo=timezone.utc)
        if polled_at >= cutoff:
            return True, None
        try:
            return False, Tick.model_validate(raw)
        except Exception:
            return False, None
    except Exception:
        return True, None  # malformed: keep to avoid silent data loss


def _prune_and_downsample(pair: str, cutoff: datetime, config: AppConfig) -> None:
    """Rewrite the hot file keeping only ticks at or after cutoff.

    Ticks that fall outside the retention window are aggregated into warm
    candles before being discarded so no price data is lost.
    """
    path = _hot_path(config.data_dir, pair)
    if not path.exists():
        return

    kept: list[str] = []
    pruned_ticks: list[Tick] = []

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            keep, tick = _classify_line(line, cutoff)
            if keep:
                kept.append(line)
            elif tick is not None:
                pruned_ticks.append(tick)

    if pruned_ticks:
        existing = read_warm_candles(pair, config)
        write_warm_candles(_merge_into_candles(existing, pruned_ticks), pair, config)

    tmp_path = path.with_suffix(".ndjson.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            for line in kept:
                fh.write(line + "\n")
        os.replace(tmp_path, path)
    except Exception:
        logger.exception("Failed to prune hot tier for %s", pair)
        tmp_path.unlink(missing_ok=True)


# ── Cold tier ─────────────────────────────────────────────────────────────────


def _cold_path(data_dir: str, pair: str) -> Path:
    return Path(data_dir) / pair / "cold.json"


def read_cold_months(pair: str, config: AppConfig) -> list[ColdMonth]:
    path = _cold_path(config.data_dir, pair)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    months: list[ColdMonth] = []
    for entry in raw:
        try:
            months.append(ColdMonth.model_validate(entry))
        except Exception as exc:
            logger.warning("Skipping malformed cold entry for %s: %s", pair, exc)
    return months


def write_cold_months(months: list[ColdMonth], pair: str, config: AppConfig) -> None:
    path = _cold_path(config.data_dir, pair)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump([m.model_dump(mode="json") for m in months], fh)
        os.replace(tmp_path, path)
    except Exception:
        logger.exception("Failed to write cold months for %s", pair)
        tmp_path.unlink(missing_ok=True)


def recompute_cold_tier(pair: str, config: AppConfig) -> None:
    """Incorporate new warm candles into the cold-tier monthly archive.

    Each warm candle is counted exactly once: candles with an hour already
    recorded in last_candle_hour for that month are skipped.
    """
    warm = read_warm_candles(pair, config)
    if not warm:
        return

    month_map: dict[str, ColdMonth] = {
        m.month: m for m in read_cold_months(pair, config)
    }

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
            n = len(new)
            total = old.candle_count + n
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

    write_cold_months(
        sorted(month_map.values(), key=lambda m: m.month), pair, config
    )
