"""Rule 12 — Cross-asset: Enhanced Lead-Lag with Dynamic Correlation Threshold."""
from __future__ import annotations

import math
import numpy as np
import logging

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle

# Constants for the original rule
MIN_CANDLES = 20  # Minimum warm candles per asset for reliable correlation and dynamic threshold calculation
MAX_LAG = 3  # Maximum lead time in hours to consider
LEAD_THRESHOLD = 0.01  # A's k-hour cumulative return must exceed this to signal

# New constants for dynamic correlation threshold
CORRELATION_WINDOW = 5  # Number of hourly candles for rolling correlation calculation (e.g., 5 hours)
THRESHOLD_LOOKBACK_PERIOD = 10  # Number of rolling correlation values to calculate mean/std dev (e.g., 10 hours)
STD_DEV_MULTIPLIER = 1.0  # Multiplier for standard deviation (e.g., mean + 1.0 * std dev)

# Derived minimum number of 1-period returns required after alignment and shifting
# This is the minimum length of `r_a_for_corr` and `r_b_lagged_for_corr`
# to produce enough rolling correlation values for the dynamic threshold calculation.
# We need `THRESHOLD_LOOKBACK_PERIOD` values of rolling correlation.
# Each rolling correlation value needs `CORRELATION_WINDOW` 1-period returns.
# So, `(length of returns series) - CORRELATION_WINDOW + 1 >= THRESHOLD_LOOKBACK_PERIOD`
# => `length of returns series >= THRESHOLD_LOOKBACK_PERIOD + CORRELATION_WINDOW - 1`
MIN_ALIGNED_RETURNS_FOR_THRESHOLD = THRESHOLD_LOOKBACK_PERIOD + CORRELATION_WINDOW - 1

# The minimum number of original warm candles required for a pair.
# This ensures that after calculating 1-period returns (losing 1 candle)
# and after shifting for lag `k` (losing `k` candles),
# we still have `MIN_ALIGNED_RETURNS_FOR_THRESHOLD` data points.
# Max `k` is `MAX_LAG`.
# So, `MIN_CANDLES_REQUIRED = 1 (for 1-period return) + MAX_LAG + MIN_ALIGNED_RETURNS_FOR_THRESHOLD - 1`
# Example: `1 + 3 + (10 + 5 - 1) - 1 = 4 + 14 - 1 = 17`.
# The existing `MIN_CANDLES = 20` is sufficient for these requirements.

# ── Pair cache (refreshed when warm tier changes, i.e. ~hourly) ───────────────
# Stores (A, B, best_lag, current_correlation, current_dynamic_threshold)
_cached_pairs: list[tuple[str, str, int, float, float]] = []
_cache_key: str = ""

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _calculate_1period_returns_series(closes: list[float]) -> np.ndarray:
    """Calculate 1-period returns series from a list of close prices."""
    arr = np.array(closes, dtype=np.float64)
    if len(arr) < 2:
        return np.array([])
    
    # Handle division by zero for returns by setting to NaN
    denominator = np.where(arr[:-1] != 0, arr[:-1], np.nan)
    returns = (arr[1:] - arr[:-1]) / denominator
    
    # Filter out NaNs to ensure subsequent calculations work with valid numbers
    # If all returns are NaN, this will return an empty array.
    return returns[~np.isnan(returns)]

def _calculate_lag_period_return(closes: list[float], lag_period: int) -> float | None:
    """Calculate the cumulative return over a specific lag period for the most recent data."""
    if len(closes) < lag_period + 1:
        return None
    start_price = closes[-lag_period - 1]
    end_price = closes[-1]
    if start_price == 0:  # Avoid division by zero
        return None
    return (end_price - start_price) / start_price

def _rolling_correlation(r_a: np.ndarray, r_b_lagged: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling Pearson correlation between two return series.
    r_a and r_b_lagged are expected to be already aligned (r_A[t] vs r_B[t+k]) and of equal length.
    """
    if len(r_a) < window or len(r_b_lagged) < window:
        return np.array([])

    correlations = []
    # Loop to calculate correlation for each window
    for i in range(len(r_a) - window + 1):
        sub_r_a = r_a[i : i + window]
        sub_r_b = r_b_lagged[i : i + window]

        # Check for sufficient variance within the window to calculate correlation
        if np.std(sub_r_a) == 0 or np.std(sub_r_b) == 0:
            correlations.append(np.nan)
            continue

        c = np.corrcoef(sub_r_a, sub_r_b)[0, 1]
        if math.isfinite(c):
            correlations.append(c)
        else:
            correlations.append(np.nan)
            
    return np.array(correlations)

def _dynamic_correlation_threshold(
    rolling_corr_series: np.ndarray, lookback_period: int, std_multiplier: float
) -> np.ndarray:
    """
    Calculate a dynamic correlation threshold based on historical rolling correlations.
    Threshold = mean + std_multiplier * std_dev of the rolling correlation series over a lookback period.
    Returns a series of thresholds.
    """
    if len(rolling_corr_series) < lookback_period:
        return np.array([])

    thresholds = []
    for i in range(len(rolling_corr_series) - lookback_period + 1):
        window_corrs = rolling_corr_series[i : i + lookback_period]
        
        # Filter out NaNs from the window, but ensure enough valid data remains
        valid_corrs = window_corrs[~np.isnan(window_corrs)]
        if len(valid_corrs) < lookback_period * 0.5: # Require at least 50% valid data points
            thresholds.append(np.nan)
            continue

        mean_corr = np.mean(valid_corrs)
        std_corr = np.std(valid_corrs)
        threshold = mean_corr + std_multiplier * std_corr
        thresholds.append(threshold)
        
    return np.array(thresholds)

# ── Pair detection (cached) ───────────────────────────────────────────────────

def _detect_and_analyze_pairs(
    asset_warm_data: dict[str, list[WarmCandle]],
) -> list[tuple[str, str, int, float, float]]:
    """
    Detects lead-lag pairs with dynamic correlation threshold.
    Returns a list of (leader_asset, follower_asset, best_lag, current_corr, current_dynamic_threshold).
    """
    pairs: list[tuple[str, str, int, float, float]] = []
    assets = list(asset_warm_data.keys())

    for i, a in enumerate(assets):
        closes_a = [c.close for c in asset_warm_data[a]]
        if len(closes_a) < MIN_CANDLES:
            continue
        returns_a_1period = _calculate_1period_returns_series(closes_a)
        
        # Ensure enough 1-period returns for asset A after calculation
        if len(returns_a_1period) < MIN_CANDLES - 1:
            continue

        for b in assets:
            if b == a:
                continue

            closes_b = [c.close for c in asset_warm_data[b]]
            if len(closes_b) < MIN_CANDLES:
                continue
            returns_b_1period = _calculate_1period_returns_series(closes_b)
            
            # Ensure enough 1-period returns for asset B after calculation
            if len(returns_b_1period) < MIN_CANDLES - 1:
                continue

            best_candidate: tuple[int, float, float] | None = None # (lag, current_corr, current_dynamic_threshold)

            for k in range(1, MAX_LAG + 1):
                # Ensure returns_b_1period is long enough to be shifted by k
                if len(returns_b_1period) <= k:
                    continue

                # Align returns_a_1period (leader) with returns_b_1period (follower shifted by k)
                # `r_a_for_corr` corresponds to `r_A[t]`
                # `r_b_lagged_for_corr` corresponds to `r_B[t+k]`
                # Both series will have length `len(original_1period_returns) - k`.
                
                # We need `len(original_1period_returns) - k` to be at least `MIN_ALIGNED_RETURNS_FOR_THRESHOLD`.
                len_after_shift_a = len(returns_a_1period) - k
                len_after_shift_b = len(returns_b_1period) - k

                if len_after_shift_a < MIN_ALIGNED_RETURNS_FOR_THRESHOLD or \
                   len_after_shift_b < MIN_ALIGNED_RETURNS_FOR_THRESHOLD:
                    continue
                
                # Trim the series for correlation calculation
                r_a_for_corr = returns_a_1period[:len_after_shift_a]
                r_b_lagged_for_corr = returns_b_1period[k:k + len_after_shift_a] # Take `len_after_shift_a` elements from `k`
                
                if len(r_a_for_corr) != len(r_b_lagged_for_corr):
                    logger.error(
                        f"Length mismatch for correlation {a}-{b} at lag {k}. "
                        f"A returns: {len(r_a_for_corr)}, B lagged returns: {len(r_b_lagged_for_corr)}"
                    )
                    continue
                
                # Calculate rolling correlation
                rolling_corr_k = _rolling_correlation(r_a_for_corr, r_b_lagged_for_corr, CORRELATION_WINDOW)
                
                if len(rolling_corr_k) < THRESHOLD_LOOKBACK_PERIOD:
                    continue # Not enough rolling correlations to compute a dynamic threshold

                # Calculate dynamic threshold
                dynamic_thresh_k_series = _dynamic_correlation_threshold(
                    rolling_corr_k, THRESHOLD_LOOKBACK_PERIOD, STD_DEV_MULTIPLIER
                )
                
                if not dynamic_thresh_k_series.size or np.isnan(dynamic_thresh_k_series[-1]):
                    continue # No valid dynamic threshold at the current point

                current_correlation = rolling_corr_k[-1]
                current_dynamic_threshold = dynamic_thresh_k_series[-1]

                if math.isfinite(current_correlation) and current_correlation > current_dynamic_threshold:
                    if best_candidate is None or current_correlation > best_candidate[1]:
                        best_candidate = (k, current_correlation, current_dynamic_threshold)
            
            if best_candidate:
                pairs.append((a, b, best_candidate[0], best_candidate[1], best_candidate[2]))
    return pairs


def _get_pairs(data: MarketData) -> list[tuple[str, str, int, float, float]]:
    global _cached_pairs, _cache_key

    # Cache key: latest warm candle hour for each pair (changes once per hour)
    key_parts = []
    for pair, pd in sorted(data.items()):
        if pd.warm:
            key_parts.append(f"{pair}:{pd.warm[-1].hour.isoformat()}")
    key = "|".join(key_parts)

    if key == _cache_key:
        return _cached_pairs

    asset_warm_data: dict[str, list[WarmCandle]] = {}
    for pair, pd in data.items():
        # Only include assets with sufficient warm data for analysis
        if len(pd.warm) >= MIN_CANDLES:
            asset_warm_data[pair] = pd.warm

    _cached_pairs = _detect_and_analyze_pairs(asset_warm_data)
    _cache_key = key

    if _cached_pairs:
        logger_pairs = [
            (a, b, k, round(c, 2), round(t, 2)) for a, b, k, c, t in _cached_pairs
        ]
        logger.debug("Lead-lag pairs detected: %s", logger_pairs)
    else:
        logger.debug("No lead-lag pairs detected.")

    return _cached_pairs


# ── Signal generation ─────────────────────────────────────────────────────────


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    pairs = _get_pairs(data)
    if not pairs:
        return []

    signals: list[BuySignal | SellSignal] = []
    seen_targets: set[str] = set()

    for a, b, lag, current_correlation, current_dynamic_threshold in pairs:
        if b in seen_targets:
            continue

        pd_a = data.get(a)
        pd_b = data.get(b)
        if pd_a is None or pd_b is None:
            continue
        
        # Ensure enough warm candles for asset A to calculate `a_return`
        if len(pd_a.warm) < lag + 1:
            continue
        
        # Need hot data for asset B for signal price and timestamp
        if not pd_b.hot:
            continue

        closes_a = [c.close for c in pd_a.warm]
        a_return = _calculate_lag_period_return(closes_a, lag)

        # Skip if a_return could not be calculated or is not a finite number
        if a_return is None or not math.isfinite(a_return):
            continue

        ts = pd_b.hot[-1].polled_at
        price = pd_b.hot[-1].last_price

        # Signal condition: A's return is significant AND current correlation is above dynamic threshold
        if current_correlation > current_dynamic_threshold:
            if a_return > LEAD_THRESHOLD:
                seen_targets.add(b)
                signals.append(BuySignal(pair=b, timestamp=ts, price=price, confidence=current_correlation))
            elif a_return < -LEAD_THRESHOLD:
                seen_targets.add(b)
                signals.append(SellSignal(pair=b, timestamp=ts, price=price, confidence=current_correlation))

    return signals