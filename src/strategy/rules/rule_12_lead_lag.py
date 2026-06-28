"""Rule 12 — Cross-asset: lead-lag relationship detection.

For every ordered pair of assets (A, B), computes the Pearson cross-correlation
of their hourly returns at lags k = 1 … MAX_LAG using the warm tier:

    corr(r_A[t], r_B[t + k])   for k = 1, 2, …, MAX_LAG

A positive correlation > CORR_THRESHOLD at lag k means A's return today
predicts B's return k hours from now — A is the leader, B the follower.

Signal condition (for each detected leader-follower pair A → B at lag k):
    A's cumulative return over the last k warm candles > LEAD_THRESHOLD

Detected pairs are cached until the warm tier changes (once per hour) so the
O(N²) correlation scan runs at most once per warm refresh, not every second.
"""

from __future__ import annotations

import math

import numpy as np

from src.agent.models import BuySignal, PairData

RULE_ID = "lead_lag_cross_asset"

MIN_CANDLES = 20  # minimum warm candles per asset for reliable correlation
MAX_LAG = 3  # maximum lead time in hours to consider
CORR_THRESHOLD = 0.5  # minimum Pearson r to treat a lag as a real relationship
LEAD_THRESHOLD = 0.01  # A's k-hour cumulative return must exceed this to signal

MarketData = dict[str, PairData]

# ── Pair cache (refreshed when warm tier changes, i.e. ~hourly) ───────────────
_cached_pairs: list[tuple[str, str, int, float]] = []  # (A, B, lag, corr)
_cache_key: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _returns(closes: list[float]) -> np.ndarray:
    arr = np.array(closes, dtype=np.float64)
    return (arr[1:] - arr[:-1]) / np.where(arr[:-1] != 0, arr[:-1], 1.0)


def _best_lag_corr(r_a: np.ndarray, r_b: np.ndarray) -> tuple[int, float]:
    """Return (best_lag, corr) maximising corr(r_A[t], r_B[t+k]) over k."""
    n = min(len(r_a), len(r_b))
    best_lag, best_corr = 0, 0.0
    for k in range(1, MAX_LAG + 1):
        if n - k < 5:
            break
        c = float(np.corrcoef(r_a[: n - k], r_b[k:n])[0, 1])
        if math.isfinite(c) and c > best_corr:
            best_corr, best_lag = c, k
    return best_lag, best_corr


# ── Pair detection (cached) ───────────────────────────────────────────────────


def _detect_pairs(
    asset_returns: dict[str, np.ndarray],
) -> list[tuple[str, str, int, float]]:
    pairs: list[tuple[str, str, int, float]] = []
    assets = list(asset_returns.keys())
    for i, a in enumerate(assets):
        for b in assets:
            if b == a:
                continue
            lag, corr = _best_lag_corr(asset_returns[a], asset_returns[b])
            if corr > CORR_THRESHOLD:
                pairs.append((a, b, lag, corr))
    return pairs


def _get_pairs(data: MarketData) -> list[tuple[str, str, int, float]]:
    global _cached_pairs, _cache_key

    # Cache key: latest warm candle hour for each pair (changes once per hour)
    key = "|".join(
        f"{pair}:{pd.warm[-1].hour.isoformat()}"
        for pair, pd in sorted(data.items())
        if pd.warm
    )
    if key == _cache_key:
        return _cached_pairs

    asset_returns: dict[str, np.ndarray] = {}
    for pair, pd in data.items():
        if len(pd.warm) >= MIN_CANDLES:
            asset_returns[pair] = _returns([c.close for c in pd.warm])

    _cached_pairs = _detect_pairs(asset_returns)
    _cache_key = key

    if _cached_pairs:
        logger_pairs = [(a, b, k, round(c, 2)) for a, b, k, c in _cached_pairs]
        import logging

        logging.getLogger(__name__).debug("Lead-lag pairs detected: %s", logger_pairs)

    return _cached_pairs


# ── Signal generation ─────────────────────────────────────────────────────────


def lead_lag_cross_asset(data: MarketData) -> list[BuySignal]:
    pairs = _get_pairs(data)
    if not pairs:
        return []

    signals: list[BuySignal] = []
    seen_targets: set[str] = set()  # emit at most one signal per target asset

    for a, b, lag, corr in pairs:
        if b in seen_targets:
            continue

        pd_a = data.get(a)
        pd_b = data.get(b)
        if pd_a is None or pd_b is None:
            continue
        if len(pd_a.warm) < lag + 1 or not pd_b.hot:
            continue

        # A's cumulative return over the last `lag` candles
        closes_a = [c.close for c in pd_a.warm]
        denom = closes_a[-lag - 1]
        if denom == 0:
            continue
        a_return = (closes_a[-1] - denom) / denom

        if a_return > LEAD_THRESHOLD:
            seen_targets.add(b)
            signals.append(
                BuySignal(
                    pair=b,
                    rule_id=RULE_ID,
                    timestamp=pd_b.hot[-1].polled_at,
                    price=pd_b.hot[-1].last_price,
                    confidence=corr,
                )
            )

    return signals
