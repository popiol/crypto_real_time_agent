"""Kraken public REST API client.

Fetches Ticker and Depth (order book) data for a list of pairs.
Handles rate-limit responses with exponential backoff.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx

from src.agent.models import AppConfig, OrderBook, OrderBookLevel, Tick

logger = logging.getLogger(__name__)

_TICKER_URL = "https://api.kraken.com/0/public/Ticker"
_DEPTH_URL = "https://api.kraken.com/0/public/Depth"

# Kraken returns this error string on rate limiting
_RATE_LIMIT_ERROR = "EGeneral:Too many requests"


def _parse_order_book(raw: dict) -> OrderBook:
    def parse_levels(entries: list) -> list[OrderBookLevel]:
        return [
            OrderBookLevel(price=float(e[0]), volume=float(e[1]), timestamp=int(e[2]))
            for e in entries
        ]

    return OrderBook(
        asks=parse_levels(raw["asks"]),
        bids=parse_levels(raw["bids"]),
    )


def _fetch_with_backoff(
    client: httpx.Client,
    url: str,
    params: dict,
    backoff_initial: float,
    backoff_max: float,
) -> dict:
    """GET url with params, retrying on rate-limit errors using exponential backoff."""
    delay = backoff_initial
    while True:
        response = client.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        errors: list[str] = data.get("error", [])
        if _RATE_LIMIT_ERROR in errors:
            logger.warning("Rate limited by Kraken, backing off for %.1fs", delay)
            time.sleep(delay)
            delay = min(delay * 2, backoff_max)
            continue

        if errors:
            raise RuntimeError(f"Kraken API error: {errors}")

        return data["result"]


def collect(config: AppConfig) -> list[Tick]:
    """Fetch one snapshot for every configured pair.

    Returns a list of Tick objects, one per pair. Pairs that fail to parse
    are skipped with a warning rather than aborting the whole cycle.
    """
    pairs_param = ",".join(config.pairs)
    polled_at = datetime.now(timezone.utc)

    with httpx.Client() as client:
        # --- Ticker: one request for all pairs ---
        ticker_result = _fetch_with_backoff(
            client,
            _TICKER_URL,
            {"pair": pairs_param},
            config.backoff_initial_seconds,
            config.backoff_max_seconds,
        )

        ticks: list[Tick] = []

        for pair in config.pairs:
            # Kraken may return the pair under an internal alias; find the matching key
            ticker_key = _find_key(ticker_result, pair)
            if ticker_key is None:
                logger.warning("Pair %s not found in Ticker response, skipping", pair)
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
                logger.warning("Failed to parse Ticker for %s: %s", pair, exc)
                continue

            # --- Depth: separate request per pair (required by Kraken API) ---
            order_book: OrderBook | None = None
            try:
                depth_result = _fetch_with_backoff(
                    client,
                    _DEPTH_URL,
                    {"pair": pair, "count": 5},
                    config.backoff_initial_seconds,
                    config.backoff_max_seconds,
                )
                depth_key = next(iter(depth_result))
                order_book = _parse_order_book(depth_result[depth_key])
            except Exception as exc:
                logger.warning("Failed to fetch Depth for %s: %s", pair, exc)

            ticks.append(
                Tick(
                    pair=pair,
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


def _find_key(result: dict, pair: str) -> str | None:
    """Find the result key that corresponds to the requested pair.

    Kraken returns pairs under legacy internal names, e.g.:
      XBTUSD  → XXBTZUSD  (leading X added to crypto asset, Z added to fiat)
      ETHUSD  → XETHZUSD

    We normalise each result key by stripping those prefixes before comparing.
    """
    if pair in result:
        return pair
    for key in result:
        if _normalise_pair(key) == pair:
            return key
    return None


def _normalise_pair(key: str) -> str:
    """Strip Kraken's legacy X/Z asset prefixes from an internal pair name.

    Examples:
      XXBTZUSD → XBTUSD
      XETHZUSD → ETHUSD
    """
    s = key
    # Strip one leading 'X' if the key would otherwise be too long
    if s.startswith("X") and len(s) >= 7:
        s = s[1:]
    # Strip 'Z' fiat prefix: find 'Z' followed by exactly 3 uppercase letters at the end
    if len(s) >= 6:
        for i in range(1, len(s) - 2):
            if s[i] == "Z" and s[i + 1 :].isupper() and len(s[i + 1 :]) == 3:
                s = s[:i] + s[i + 1 :]
                break
    return s
