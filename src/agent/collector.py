"""Kraken data collector.

Fetches Ticker and Depth snapshots for all configured pairs using KrakenClient.
Handles rate-limit responses with exponential backoff.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, TypeVar

from src.agent.models import AppConfig, OrderBook, OrderBookLevel, Tick
from src.kraken import KrakenClient, KrakenError

logger = logging.getLogger(__name__)

_RATE_LIMIT_ERROR = "EGeneral:Too many requests"

_T = TypeVar("_T")


def _with_backoff(
    fn: Callable[[], _T], backoff_initial: float, backoff_max: float
) -> _T:
    """Call fn(), retrying on Kraken rate-limit errors with exponential backoff."""
    delay = backoff_initial
    while True:
        try:
            return fn()
        except KrakenError as exc:
            errors: list = exc.args[0] if exc.args else []
            if isinstance(errors, list) and any(
                _RATE_LIMIT_ERROR in str(e) for e in errors
            ):
                logger.warning("Rate limited by Kraken, backing off for %.1fs", delay)
                time.sleep(delay)
                delay = min(delay * 2, backoff_max)
            else:
                raise


def _parse_order_book(raw: dict) -> OrderBook:
    def parse_levels(entries: list) -> list[OrderBookLevel]:
        return [
            OrderBookLevel(price=float(e[0]), volume=float(e[1]), timestamp=int(e[2]))
            for e in entries
        ]

    return OrderBook(asks=parse_levels(raw["asks"]), bids=parse_levels(raw["bids"]))


def _find_key(result: dict, altname: str) -> str | None:
    """Map a Kraken Ticker response key back to the requested altname.

    Kraken may key its response by a legacy internal name (e.g. XXBTZUSD
    instead of XBTUSD). We normalise both sides before comparing.
    """
    if altname in result:
        return altname
    for key in result:
        if _normalise_pair(key) == altname:
            return key
    return None


def _normalise_pair(key: str) -> str:
    """Strip Kraken legacy X/Z asset prefixes from an internal pair name."""
    s = key
    if s.startswith("X") and len(s) >= 7:
        s = s[1:]
    if len(s) >= 6:
        for i in range(1, len(s) - 2):
            if s[i] == "Z" and s[i + 1 :].isupper() and len(s[i + 1 :]) == 3:
                s = s[:i] + s[i + 1 :]
                break
    return s


def collect(config: AppConfig) -> list[Tick]:
    """Fetch one snapshot for every configured pair.

    Returns a list of Tick objects, one per pair. Pairs that fail to parse
    are skipped with a warning rather than aborting the whole cycle.
    """
    polled_at = datetime.now(timezone.utc)
    kraken = KrakenClient()

    usd_pairs = _with_backoff(
        kraken._load_usd_pairs,
        config.backoff_initial_seconds,
        config.backoff_max_seconds,
    )

    if config.pairs:
        altname_index = {info.altname: info for info in usd_pairs.values()}
        pair_infos = [altname_index[a] for a in config.pairs if a in altname_index]
    else:
        pair_infos = list(usd_pairs.values())

    pairs_param = ",".join(p.altname for p in pair_infos)
    ticker_result = _with_backoff(
        lambda: kraken._public_get("Ticker", {"pair": pairs_param}),
        config.backoff_initial_seconds,
        config.backoff_max_seconds,
    )

    ticks: list[Tick] = []
    for pair_info in pair_infos:
        ticker_key = _find_key(ticker_result, pair_info.altname)
        if ticker_key is None:
            logger.warning(
                "Pair %s not found in Ticker response, skipping", pair_info.altname
            )
            continue

        t = ticker_result[ticker_key]
        try:
            bid_price = float(t["b"][0])
            bid_volume = float(t["b"][1])
            ask_price = float(t["a"][0])
            ask_volume = float(t["a"][1])
            last_price = float(t["c"][0])
            mid_price = (bid_price + ask_price) / 2
            spread_abs = ask_price - bid_price
            spread_rel = (spread_abs / mid_price * 100) if mid_price else 0.0
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("Failed to parse Ticker for %s: %s", pair_info.altname, exc)
            continue

        order_book: OrderBook | None = None
        try:
            altname = pair_info.altname
            depth_result = _with_backoff(
                lambda a=altname: kraken._public_get("Depth", {"pair": a, "count": 5}),
                config.backoff_initial_seconds,
                config.backoff_max_seconds,
            )
            depth_key = next(iter(depth_result))
            order_book = _parse_order_book(depth_result[depth_key])
        except Exception as exc:
            logger.warning("Failed to fetch Depth for %s: %s", pair_info.altname, exc)

        ticks.append(
            Tick(
                pair=pair_info.altname,
                polled_at=polled_at,
                last_price=last_price,
                bid_price=bid_price,
                bid_volume=bid_volume,
                ask_price=ask_price,
                ask_volume=ask_volume,
                mid_price=mid_price,
                spread_abs=spread_abs,
                spread_rel=spread_rel,
                order_book=order_book,
            )
        )

    return ticks
