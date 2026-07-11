"""Rule 12 — Cross-asset: lead-lag relationship detection, volatility-adjusted.

For every ordered pair of assets (A, B), computes the Pearson cross-correlation
of their hourly returns at lags k = 1 … MAX_LAG using the warm tier:

    corr(r_A[t], r_B[t + k])   for k = 1, 2, …, MAX_LAG

A positive correlation > CORR_THRESHOLD at lag k means A's return today
predicts B's return k hours from now — A is the leader, B the follower.

Buy signal:  A's k-hour cumulative return >  DYNAMIC_THRESHOLD → B expected to rise.
Sell signal: A's k-hour cumulative return < -DYNAMIC_THRESHOLD → B expected to fall.

The DYNAMIC_THRESHOLD is calculated as a multiple of the leading asset's
recent historical standard deviation, aiming to filter out noise.

At most one signal is emitted per target asset per cycle.

Detected pairs are cached until the warm tier changes (once per hour) so the
O(N²) correlation scan runs at most once per warm refresh, not every second.
"""

from __future__ import annotations

import math

import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# ── Configuration Constants ───────────────────────────────────────────────────
MIN_CANDLES_FOR_CORRELATION = 20  # minimum warm candles per asset for reliable correlation
MAX_LAG = 3  # maximum lead time in hours to consider
CORR_THRESHOLD = 0.5  # minimum Pearson r to treat a lag as a real relationship

# New constants for volatility-adjusted threshold
VOLATILITY_CANDLE_LOOKBACK = 20  # Number of *returns* to consider for volatility calculation (requires VOLATILITY_CANDLE_LOOKBACK + 1 candles)
VOLATILITY_MULTIPLIER = 1.5  # Multiplier for standard deviation to set the dynamic threshold

# Minimum candles required for a leader asset to calculate both its k-hour return
# and its volatility, ensuring enough data for both operations.
MIN_CANDLES_FOR_SIGNAL = max(MAX_LAG + 1, VOLATILITY_CANDLE_LOOKBACK + 1)


# ── Pair cache (refreshed when warm tier changes, i.e. ~hourly) ───────────────
_cached_pairs: list[tuple[str, str, int, float]] = []  # (A, B, lag, corr)
_cache_key: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _returns(closes: list[float]) -> np.ndarray:
    """Calculate percentage returns from a list of close prices."""
    arr = np.array(closes, dtype=np.float64)
    # Avoid division by zero by replacing zero denominators with 1.0,
    # which will result in a return of (price - 0) / 1 = price for that period,
    # effectively treating it as an invalid/missing prior price, but preventing error.
    # Given the original implementation, we stick to this approach.
    return (arr[1:] - arr[:-1]) / np.where(arr[:-1] != 0, arr[:-1], 1.0)


def _best_lag_corr(r_a: np.ndarray, r_b: np.ndarray) -> tuple[int, float]:
    """Return (best_lag, corr) maximising corr(r_A[t], r_B[t+k]) over k."""
    # Ensure sufficient data for correlation calculation
    # np.corrcoef requires at least 2 observations.
    min_observations_for_corr = 5

    n = min(len(r_a), len(r_b))
    best_lag, best_corr = 0, 0.0
    for k in range(1, MAX_LAG + 1):
        if n - k < min_observations_for_corr:
            # Not enough data for this lag
            break
        # Calculate Pearson correlation coefficient
        # np.corrcoef returns a 2x2 matrix; we want the off-diagonal element
        corr_matrix = np.corrcoef(r_a[: n - k], r_b[k:n])
        c = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

        if c > best_corr: # Only care about positive correlations for lead-lag
            best_corr, best_lag = c, k
    return best_lag, best_corr


# ── Pair detection (cached) ───────────────────────────────────────────────────


def _detect_pairs(
    asset_returns: dict[str, np.ndarray],
) -> list[tuple[str, str, int, float]]:
    """
    Detects lead-lag relationships between asset pairs based on cross-correlation
    of their returns.
    """
    pairs: list[tuple[str, str, int, float]] = []
    assets = list(asset_returns.keys())
    for i, a in enumerate(assets):
        for b in assets:
            if b == a:
                continue
            # _returns produces N-1 returns from N candles.
            # MIN_CANDLES_FOR_CORRELATION is the number of candles needed.
            # So, we need at least MIN_CANDLES_FOR_CORRELATION - 1 returns.
            if len(asset_returns[a]) < MIN_CANDLES_FOR_CORRELATION - 1 or \
               len(asset_returns[b]) < MIN_CANDLES_FOR_CORRELATION - 1:
                continue

            lag, corr = _best_lag_corr(asset_returns[a], asset_returns[b])
            if corr > CORR_THRESHOLD:
                pairs.append((a, b, lag, corr))
    return pairs


def _get_pairs(data: MarketData) -> list[tuple[str, str, int, float]]:
    """
    Retrieves lead-lag pairs, using a cache to avoid recomputing every tick.
    The cache is refreshed hourly based on the latest warm candle timestamp.
    """
    global _cached_pairs, _cache_key

    # Cache key: latest warm candle hour for each pair (changes once per hour)
    # Using sorted keys to ensure consistent cache key across runs/platforms
    key = "|".join(
        f"{pair}:{pd.warm[-1].hour.isoformat()}"
        for pair, pd in sorted(data.items())
        if pd.warm # only include pairs that have warm data
    )
    if key == _cache_key:
        return _cached_pairs

    asset_returns: dict[str, np.ndarray] = {}
    for pair, pd in data.items():
        # Need at least MIN_CANDLES_FOR_CORRELATION candles to calculate returns
        if len(pd.warm) >= MIN_CANDLES_FOR_CORRELATION:
            asset_returns[pair] = _returns([c.close for c in pd.warm])

    _cached_pairs = _detect_pairs(asset_returns)
    _cache_key = key

    # import logging
    # if _cached_pairs:
    #     logging.getLogger(__name__).debug("Lead-lag pairs detected: %s", [(a, b, k, round(c, 2)) for a, b, k, c in _cached_pairs])

    return _cached_pairs


# ── Signal generation ─────────────────────────────────────────────────────────


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates buy/sell signals based on detected lead-lag relationships
    and a volatility-adjusted threshold for the leader's price movement.
    """
    pairs = _get_pairs(data)
    if not pairs:
        return []

    signals: list[BuySignal | SellSignal] = []
    seen_targets: set[str] = set()  # To ensure at most one signal per target asset

    for a, b, lag, corr in pairs:
        if b in seen_targets:
            continue

        pd_a = data.get(a)  # Leader asset data
        pd_b = data.get(b)  # Lagging asset data
        if pd_a is None or pd_b is None:
            continue

        # Ensure enough warm candles for leader (A) for both k-hour return and volatility
        # And ensure hot data for lagging asset (B) for signal price/timestamp
        if len(pd_a.warm) < MIN_CANDLES_FOR_SIGNAL or not pd_b.hot:
            continue

        closes_a = [c.close for c in pd_a.warm]

        # 1. Calculate the leader's (A) k-hour cumulative return
        # The lookback for this return is 'lag' periods.
        # closes_a[-1] is the current close, closes_a[-lag - 1] is the close 'lag' hours ago.
        # This length check is technically redundant due to MIN_CANDLES_FOR_SIGNAL, but kept for clarity
        if len(closes_a) < lag + 1:
            continue
        denom_a_return = closes_a[-lag - 1]
        if denom_a_return == 0:
            continue # Avoid division by zero for return calculation
        a_return = (closes_a[-1] - denom_a_return) / denom_a_return

        # 2. Calculate the dynamic significance threshold based on leader's volatility
        # We need VOLATILITY_CANDLE_LOOKBACK + 1 candles for VOLATILITY_CANDLE_LOOKBACK returns.
        # We take the most recent candles for volatility calculation.
        # This length check is also redundant due to MIN_CANDLES_FOR_SIGNAL, but kept for clarity
        if len(closes_a) < VOLATILITY_CANDLE_LOOKBACK + 1:
            continue
        
        # Slice the candles needed for volatility calculation
        # e.g., if VOLATILITY_CANDLE_LOOKBACK = 20, we need closes_a[-21:] for 20 returns
        closes_a_for_volatility = closes_a[-(VOLATILITY_CANDLE_LOOKBACK + 1):]
        leader_returns_for_volatility = _returns(closes_a_for_volatility)
        
        # np.std of an empty array returns NaN. np.std of a single element array returns 0.0.
        # VOLATILITY_CANDLE_LOOKBACK ensures enough returns for a meaningful std dev.
        # If for some edge case, all prices were identical, std dev would be 0, threshold 0.
        # This is reasonable: any movement from a perfectly flat history is "significant".
        if len(leader_returns_for_volatility) == 0:
            continue # Should not happen with MIN_CANDLES_FOR_SIGNAL, but defensive
            
        leader_std_dev = np.std(leader_returns_for_volatility)
        
        # Define the dynamic threshold
        dynamic_significance_threshold = VOLATILITY_MULTIPLIER * leader_std_dev

        # 3. Generate signal if leader's movement exceeds the dynamic threshold
        ts = pd_b.hot[-1].polled_at
        price = pd_b.hot[-1].last_price

        if a_return > dynamic_significance_threshold:
            seen_targets.add(b)
            signals.append(BuySignal(pair=b, timestamp=ts, price=price, confidence=corr))
        elif a_return < -dynamic_significance_threshold:
            seen_targets.add(b)
            signals.append(SellSignal(pair=b, timestamp=ts, price=price, confidence=corr))

    return signals