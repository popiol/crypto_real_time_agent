from __future__ import annotations

import math
import logging

import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


# ── Constants for Rule 12 ─────────────────────────────────────────────────────

MIN_CANDLES = 20  # minimum warm candles per asset for reliable correlation in _detect_pairs
MAX_LAG = 3  # maximum lead time in hours to consider
CORR_THRESHOLD = 0.5  # minimum Pearson r to treat a lag as a real relationship (for pair detection)
LEAD_THRESHOLD = 0.01  # A's k-hour cumulative return must exceed this to signal

# ── New Constants for Dynamic Minimum Correlation Threshold ───────────────────

# The window size (in hourly candles) for calculating the latest rolling correlation.
# Using MIN_CANDLES ensures we have a reasonable history for the dynamic check.
CORRELATION_WINDOW = MIN_CANDLES
# Minimum absolute correlation required for a signal to be generated.
# This filters out signals during periods of weak lead-lag relationships.
MIN_ABS_CORRELATION_THRESHOLD = 0.7


# ── Pair cache (refreshed when warm tier changes, i.e. ~hourly) ───────────────
_cached_pairs: list[tuple[str, str, int, float]] = []  # (A, B, lag, corr)
_cache_key: str = ""

# Initialize logger
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _returns(closes: list[float]) -> np.ndarray:
    """Calculates percentage returns from a list of closing prices."""
    arr = np.array(closes, dtype=np.float64)
    # Avoid division by zero for initial prices
    return (arr[1:] - arr[:-1]) / np.where(arr[:-1] != 0, arr[:-1], 1.0)


def _best_lag_corr(r_a: np.ndarray, r_b: np.ndarray) -> tuple[int, float]:
    """
    Return (best_lag, corr) maximising corr(r_A[t], r_B[t+k]) over k.
    This correlation is calculated over the entire available historical returns.
    """
    n = min(len(r_a), len(r_b))
    best_lag, best_corr = 0, 0.0
    for k in range(1, MAX_LAG + 1):
        # Need at least 5 data points for a meaningful correlation
        if n - k < 5:
            break
        # Correlate r_A at time t with r_B at time t+k
        # This means r_A's window ends k periods before r_B's window ends
        c = float(np.corrcoef(r_a[: n - k], r_b[k:n])[0, 1])
        if math.isfinite(c) and c > best_corr:
            best_corr, best_lag = c, k
    return best_lag, best_corr


# ── Pair detection (cached) ───────────────────────────────────────────────────


def _detect_pairs(
    asset_returns: dict[str, np.ndarray],
) -> list[tuple[str, str, int, float]]:
    """
    Detects lead-lag pairs based on historical correlation.
    Pairs are detected if their correlation exceeds CORR_THRESHOLD.
    """
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
    """
    Retrieves or refreshes the list of lead-lag pairs.
    The list is cached and refreshed only when the warm tier data changes.
    """
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
        logger.debug("Lead-lag pairs detected: %s", logger_pairs)

    return _cached_pairs


# ── Signal generation ─────────────────────────────────────────────────────────


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on detected lead-lag relationships
    and a dynamic minimum correlation threshold.
    """
    pairs = _get_pairs(data)
    if not pairs:
        return []

    signals: list[BuySignal | SellSignal] = []
    seen_targets: set[str] = set()

    for a, b, lag, corr in pairs:
        # Ensure only one signal per target asset per cycle
        if b in seen_targets:
            continue

        pd_a = data.get(a)
        pd_b = data.get(b)
        if pd_a is None or pd_b is None:
            continue

        # Ensure enough warm data for both lead-lag calculation and dynamic correlation check
        # For 'a_return' (leader's movement): we need at least `lag + 1` warm candles.
        # For 'dynamic correlation':
        #   A (leader) needs `CORRELATION_WINDOW + lag + 1` warm candles to get `CORRELATION_WINDOW` returns
        #     that lead B's window by `lag`.
        #   B (follower) needs `CORRELATION_WINDOW + 1` warm candles to get `CORRELATION_WINDOW` returns.
        min_warm_a = max(lag + 1, CORRELATION_WINDOW + lag + 1)
        min_warm_b = CORRELATION_WINDOW + 1

        if len(pd_a.warm) < min_warm_a or len(pd_b.warm) < min_warm_b or not pd_b.hot:
            continue

        # ── NEW: Dynamic Minimum Correlation Threshold Check ──────────────────
        closes_a_warm = [c.close for c in pd_a.warm]
        closes_b_warm = [c.close for c in pd_b.warm]

        # Extract price windows for calculating the latest rolling correlation.
        # For A (leader): we need prices from (current_time - CORRELATION_WINDOW - lag)
        # to (current_time - lag - 1) to get CORRELATION_WINDOW returns.
        # This means `CORRELATION_WINDOW + 1` prices.
        # Example: if CORRELATION_WINDOW=3, lag=1, we need prices for r_A[t-4], r_A[t-3], r_A[t-2]
        # These come from prices at [t-5, t-4, t-3, t-2].
        # So, slice `[-(CORRELATION_WINDOW + lag + 1) : -lag]` from closes_a_warm.
        prices_a_for_corr = closes_a_warm[-(CORRELATION_WINDOW + lag + 1) : -lag]

        # For B (follower): we need prices from (current_time - CORRELATION_WINDOW)
        # to (current_time - 1) to get CORRELATION_WINDOW returns.
        # This means `CORRELATION_WINDOW + 1` prices.
        # Example: if CORRELATION_WINDOW=3, we need prices for r_B[t-3], r_B[t-2], r_B[t-1]
        # These come from prices at [t-4, t-3, t-2, t-1].
        # So, slice `[-(CORRELATION_WINDOW + 1) : ]` from closes_b_warm.
        prices_b_for_corr = closes_b_warm[-(CORRELATION_WINDOW + 1) :]

        # Ensure we have enough prices to form the specified number of returns
        if (len(prices_a_for_corr) < CORRELATION_WINDOW + 1 or
            len(prices_b_for_corr) < CORRELATION_WINDOW + 1):
            logger.debug(
                "Not enough data for dynamic correlation for pair %s-%s (lag %d). "
                "A prices: %d, B prices: %d. Required: A %d, B %d.",
                a, b, lag, len(prices_a_for_corr), len(prices_b_for_corr),
                CORRELATION_WINDOW + 1, CORRELATION_WINDOW + 1
            )
            continue

        returns_a_for_corr = _returns(prices_a_for_corr)
        returns_b_for_corr = _returns(prices_b_for_corr)

        # Ensure the returns arrays have the expected length
        if (len(returns_a_for_corr) < CORRELATION_WINDOW or
            len(returns_b_for_corr) < CORRELATION_WINDOW):
            logger.debug(
                "Not enough returns for dynamic correlation for pair %s-%s (lag %d). "
                "A returns: %d, B returns: %d. Required: %d.",
                a, b, lag, len(returns_a_for_corr), len(returns_b_for_corr),
                CORRELATION_WINDOW
            )
            continue

        # Calculate the current correlation for the latest window
        current_rolling_corr = float(np.corrcoef(returns_a_for_corr, returns_b_for_corr)[0, 1])

        # Apply the new dynamic minimum correlation threshold
        if not math.isfinite(current_rolling_corr) or \
           abs(current_rolling_corr) < MIN_ABS_CORRELATION_THRESHOLD:
            logger.debug(
                "Skipping signal for pair %s-%s (lag %d) due to weak dynamic correlation: %.2f (threshold %.2f)",
                a, b, lag, current_rolling_corr, MIN_ABS_CORRELATION_THRESHOLD
            )
            continue  # Skip signal if correlation is weak or unstable
        # ── END NEW ───────────────────────────────────────────────────────────

        # Calculate leading asset's movement over the lag period
        # Get the closing price of asset A at `lag` hours ago
        # (index -1 is current, -2 is 1 hour ago, etc.)
        denom = closes_a_warm[-lag - 1]
        if denom == 0:
            logger.debug("Skipping signal for pair %s-%s (lag %d) due to zero denominator in A's return calculation.", a, b, lag)
            continue
        a_return = (closes_a_warm[-1] - denom) / denom

        ts = pd_b.hot[-1].polled_at
        price = pd_b.hot[-1].last_price

        # Generate signals based on A's movement
        if a_return > LEAD_THRESHOLD:
            seen_targets.add(b)
            signals.append(BuySignal(pair=b, timestamp=ts, price=price, confidence=corr))
            logger.info(
                "BUY signal for %s: %s moved +%.2f%% over %dh (corr %.2f, dynamic_corr %.2f)",
                b, a, a_return * 100, lag, corr, current_rolling_corr
            )
        elif a_return < -LEAD_THRESHOLD:
            seen_targets.add(b)
            signals.append(SellSignal(pair=b, timestamp=ts, price=price, confidence=corr))
            logger.info(
                "SELL signal for %s: %s moved %.2f%% over %dh (corr %.2f, dynamic_corr %.2f)",
                b, a, a_return * 100, lag, corr, current_rolling_corr
            )

    return signals