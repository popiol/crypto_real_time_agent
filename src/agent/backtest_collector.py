"""Backtest collector — replaces the live Kraken collector in test mode.

Reads historical Ticker snapshots from backtest_data_dir, which contains files
partitioned as year=YYYY/month=MM/day=DD/YYYYMMDDHHmmss.json. Each file is a
Kraken Ticker API response keyed by pair altname. A companion *_bidask.json
file at the same timestamp, if present, provides order book depth data.

next_snapshot() advances a module-level cursor through the ticker files in
chronological order, returning one batch of Tick objects per call. Returns
None when exhausted.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.agent.collector import _find_key
from src.agent.models import AppConfig, OrderBook, OrderBookLevel, Tick

logger = logging.getLogger(__name__)

_files: list[Path] | None = None
_cursor: int = 0


def _get_files(data_dir: str) -> list[Path]:
    global _files
    if _files is None:
        _files = sorted(
            p for p in Path(data_dir).rglob("*.json") if "_bidask" not in p.stem
        )[-700:]
        logger.info("Backtest: %d snapshot files found in %s", len(_files), data_dir)
    return _files


def _load_order_books(ticker_path: Path) -> dict[str, OrderBook]:
    bidask_path = ticker_path.with_stem(ticker_path.stem + "_bidask")
    if not bidask_path.exists():
        return {}
    try:
        raw: dict = json.loads(bidask_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    result: dict[str, OrderBook] = {}
    for pair, ob in raw.items():
        try:
            asks = [OrderBookLevel(price=float(e[0]), volume=float(e[1]), timestamp=int(e[2])) for e in ob["asks"]]
            bids = [OrderBookLevel(price=float(e[0]), volume=float(e[1]), timestamp=int(e[2])) for e in ob["bids"]]
            result[pair] = OrderBook(asks=asks, bids=bids)
        except (KeyError, IndexError, ValueError):
            continue
    return result


def next_snapshot(config: AppConfig) -> list[Tick] | None:
    """Return ticks from the next historical snapshot, or None when exhausted."""
    global _cursor
    files = _get_files(config.backtest_data_dir)

    if _cursor >= len(files):
        return None

    path = files[_cursor]
    _cursor += 1

    try:
        polled_at = datetime.strptime(path.stem[:14], "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        logger.warning("Cannot parse timestamp from %s; skipping", path.name)
        return []

    try:
        ticker_result: dict = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return []

    order_books = _load_order_books(path)
    pair_names = config.pairs if config.pairs else list(ticker_result.keys())

    ticks: list[Tick] = []
    for altname in pair_names:
        key = _find_key(ticker_result, altname)
        if key is None:
            continue
        t = ticker_result[key]
        try:
            bid_price = float(t["b"][0])
            bid_volume = float(t["b"][1])
            ask_price = float(t["a"][0])
            ask_volume = float(t["a"][1])
            last_price = float(t["c"][0])
            volume_24h = float(t["v"][1])
            mid_price = (bid_price + ask_price) / 2
            spread_abs = ask_price - bid_price
            spread_rel = (spread_abs / mid_price * 100) if mid_price else 0.0
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("Failed to parse %s in %s: %s", altname, path.name, exc)
            continue

        ticks.append(
            Tick(
                pair=altname,
                polled_at=polled_at,
                last_price=last_price,
                volume_24h=volume_24h,
                bid_price=bid_price,
                bid_volume=bid_volume,
                ask_price=ask_price,
                ask_volume=ask_volume,
                mid_price=mid_price,
                spread_abs=spread_abs,
                spread_rel=spread_rel,
                order_book=order_books.get(altname) or order_books.get(key),
            )
        )

    return ticks
