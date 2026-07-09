from __future__ import annotations

import math
import statistics
import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle


# ── Parameters for Lead-Lag with Adaptive Volatility and Mean Reversion Filter ──
CORRELATION_WINDOW = 60  # Number of bars for correlation calculation
MAX_LAG_SCAN = 5         # Maximum lead time in hours to consider for lead-lag detection
VOLATILITY_PERIOD = 14   # Period for ATR calculation for leading asset
MA_PERIOD = 10           # Period for short-term MA for lagging asset
VOLATILITY_MULTIPLIER = 2.0  # How many ATRs define 'significant' movement
MA_DEVIATION_THRESHOLD = 0.005 # Percentage deviation from MA for mean-reversion filter (e.g., 0.5%)
MIN_CORRELATION_THRESHOLD = 0.6 # Minimum absolute Pearson r to consider a lead-lag relationship

# Minimum number of warm candles required to perform correlation and subsequent indicator calculations.
# This ensures enough data for returns, ATR, and SMA for the longest lookback period.
# CORRELATION_WINDOW + MAX_LAG_SCAN + 1 is for returns for correlation (r_a, r_b)
# VOLATILITY_PERIOD + 1 is for ATR
# MA_PERIOD is for SMA
MIN_CANDLES_FOR_DATA = max(CORRELATION_WINDOW + MAX_LAG_SCAN + 1, VOLATILITY_PERIOD + 1, MA_PERIOD)


# ── Pair cache (refreshed when warm tier changes, i.e. ~hourly) ───────────────
_cached_pairs: list[tuple[str, str, int, float]] = []  # (A, B, lag, corr)
_cache_key: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _returns(closes: list[float]) -> np.ndarray:
    """Calculates percentage returns from a list of closing prices."""
    arr = np.array(closes, dtype=np.float64)
    # Avoid division by zero by replacing zero prices with 1.0 (or a small epsilon)
    # This ensures returns are calculable but might skew results for assets with actual zero prices.
    # In practice, asset prices are rarely zero.
    return (arr[1:] - arr[:-1]) / np.where(arr[:-1] != 0, arr[:-1], 1.0)


def _calculate_true_range(high: float, low: float, prev_close: float) -> float:
    """Calculates the True Range for a single candle."""
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def _calculate_atr(candles: list[WarmCandle], period: int) -> float | None:
    """Calculates the Average True Range (ATR) for a list of WarmCandles."""
    # Need `period` candles for the SMA of TR, plus one for the first prev_close
    if len(candles) < period + 1:
        return None

    true_ranges = []
    # Start from the second candle to have a previous close
    for i in range(1, len(candles)):
        prev_close = candles[i-1].close
        current_high = candles[i].high
        current_low = candles[i].low
        true_ranges.append(_calculate_true_range(current_high, current_low, prev_close))

    # Calculate SMA of the last `period` true ranges
    if len(true_ranges) < period:
        return None
    
    return statistics.mean(true_ranges[-period:])


def _calculate_sma(prices: list[float], period: int) -> float | None:
    """Calculates the Simple Moving Average (SMA) for a list of prices."""
    if len(prices) < period:
        return None
    return statistics.mean(prices[-period:])


def _best_lag_corr_abs(
    r_a_all: np.ndarray, r_b_all: np.ndarray, max_lag_to_scan: int, correlation_window: int
) -> tuple[int, float]:
    """
    Return (best_lag, corr) maximising ABS(corr(r_A[t], r_B[t+k])) over k.
    Uses `correlation_window` data points for the Pearson correlation.
    `r_a_all` and `r_b_all` are arrays of returns.
    """
    best_lag, best_corr_val = 0, 0.0  # best_corr_val stores the actual correlation, not its absolute
    max_abs_corr = 0.0

    # Ensure enough data for the correlation window plus any lag
    if len(r_a_all) < correlation_window + max_lag_to_scan or \
       len(r_b_all) < correlation_window: # r_b_all only needs correlation_window elements at the end
        return 0, 0.0

    for k in range(1, max_lag_to_scan + 1):
        # r_A is the leading asset, its returns are from `t` (earlier)
        # r_B is the lagging asset, its returns are from `t + k` (later)
        # To calculate corr(r_A[t], r_B[t+k]) for the most recent `correlation_window` period:
        # We need r_A's returns ending `k` periods ago, and r_B's returns ending now.
        r_a_slice = r_a_all[-(correlation_window + k) : -k]
        r_b_slice = r_b_all[-correlation_window :]

        # Need at least 2 data points for meaningful correlation
        if len(r_a_slice) < 2 or len(r_b_slice) < 2:
            continue
        
        # Check for standard deviation == 0 to avoid NaNs in correlation if one series is constant
        if np.std(r_a_slice) == 0 or np.std(r_b_slice) == 0:
            continue

        c = float(np.corrcoef(r_a_slice, r_b_slice)[0, 1])
        if math.isfinite(c) and abs(c) > max_abs_corr:
            max_abs_corr, best_corr_val, best_lag = abs(c), c, k
    return best_lag, best_corr_val


# ── Pair detection (cached) ───────────────────────────────────────────────────

def _detect_pairs(
    asset_returns: dict[str, np.ndarray],
    min_correlation_threshold: float,
    max_lag_to_scan: int,
    correlation_window: int
) -> list[tuple[str, str, int, float]]:
    """Detects lead-lag pairs based on lagged correlation."""
    pairs: list[tuple[str, str, int, float]] = []
    assets = list(asset_returns.keys())
    for i, a in enumerate(assets):
        for b in assets:
            if b == a:
                continue
            lag, corr = _best_lag_corr_abs(asset_returns[a], asset_returns[b], max_lag_to_scan, correlation_window)
            if abs(corr) > min_correlation_threshold:
                pairs.append((a, b, lag, corr))
    return pairs


def _get_pairs(data: MarketData) -> list[tuple[str, str, int, float]]:
    """
    Retrieves or calculates lead-lag pairs. Caches results until warm tier data changes.
    """
    global _cached_pairs, _cache_key

    # Cache key: latest warm candle hour for each pair (changes once per hour)
    # This ensures correlation is re-calculated only when new hourly data is available.
    key = "|".join(
        f"{pair}:{pd.warm[-1].hour.isoformat()}"
        for pair, pd in sorted(data.items())
        if pd.warm
    )
    if key == _cache_key:
        return _cached_pairs

    asset_returns: dict[str, np.ndarray] = {}
    for pair, pd in data.items():
        # Ensure enough warm candles for the longest lookback for returns calculation.
        # _returns will produce len(closes) - 1 returns.
        # _best_lag_corr_abs requires r_a_all to be at least CORRELATION_WINDOW + MAX_LAG_SCAN returns long.
        # So, pd.warm must be at least CORRELATION_WINDOW + MAX_LAG_SCAN + 1 candles long.
        if len(pd.warm) >= MIN_CANDLES_FOR_DATA:
            asset_returns[pair] = _returns([c.close for c in pd.warm])
    
    _cached_pairs = _detect_pairs(asset_returns, MIN_CORRELATION_THRESHOLD, MAX_LAG_SCAN, CORRELATION_WINDOW)
    _cache_key = key

    # Optional logging for detected pairs
    # import logging
    # if _cached_pairs:
    #     logging.getLogger(__name__).debug("Lead-lag pairs detected: %s", [(a, b, k, round(c, 2)) for a, b, k, c in _cached_pairs])

    return _cached_pairs


# ── Signal generation ─────────────────────────────────────────────────────────

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on the Lead-Lag with Adaptive Volatility and Mean Reversion Filter rule.
    """
    pairs = _get_pairs(data)
    if not pairs:
        return []

    signals: list[BuySignal | SellSignal] = []
    seen_targets: set[str] = set() # Ensure only one signal per target asset per cycle

    for a, b, lag, corr in pairs:
        if b in seen_targets:
            continue

        pd_a = data.get(a)
        pd_b = data.get(b)
        if pd_a is None or pd_b is None:
            continue

        # Check data sufficiency for leading asset (A) calculations
        # Needs `lag + 1` candles for price change.
        # Needs `VOLATILITY_PERIOD + 1` candles for ATR.
        min_candles_a = max(lag + 1, VOLATILITY_PERIOD + 1)
        if len(pd_a.warm) < min_candles_a:
            continue

        # Check data sufficiency for lagging asset (B) calculations
        # Needs `MA_PERIOD` candles for SMA.
        min_candles_b = MA_PERIOD
        if len(pd_b.warm) < min_candles_b:
            continue
        
        # Also need hot data for current price and timestamp for the signal itself
        if not pd_b.hot:
            continue

        # 1. Calculate recent price change for leading asset over `lag` periods
        closes_a = [c.close for c in pd_a.warm]
        leading_asset_past_price = closes_a[-(lag + 1)] # Price `lag` bars ago
        if leading_asset_past_price == 0:
            continue
        leading_asset_current_price = closes_a[-1]
        leading_asset_price_change = (leading_asset_current_price - leading_asset_past_price) / leading_asset_past_price

        # 2. Calculate volatility (ATR) for leading asset
        leading_asset_volatility = _calculate_atr(pd_a.warm, VOLATILITY_PERIOD)
        if leading_asset_volatility is None:
            continue

        # 3. Define adaptive threshold for significant movement
        # The threshold is a percentage movement relative to the price `lag` bars ago.
        if leading_asset_past_price == 0: # Defensive check
            continue
        significant_movement_threshold = VOLATILITY_MULTIPLIER * leading_asset_volatility / leading_asset_past_price
        
        # 4. Calculate short-term Moving Average for lagging asset
        closes_b = [c.close for c in pd_b.warm]
        lagging_asset_ma = _calculate_sma(closes_b, MA_PERIOD)
        if lagging_asset_ma is None:
            continue

        # 5. Calculate lagging asset price deviation from its MA
        lagging_asset_current_price_warm = closes_b[-1] # Use the most recent warm candle close for deviation
        if lagging_asset_ma == 0: # Defensive check
            continue
        lagging_asset_price_deviation = (lagging_asset_current_price_warm - lagging_asset_ma) / lagging_asset_ma

        # Get current timestamp and price for the signal
        ts = pd_b.hot[-1].polled_at
        price = pd_b.hot[-1].last_price

        # 6. Generate Buy Signal
        # (lagged_correlation > 0 AND leading_asset_price_change > significant_movement_threshold AND lagging_asset_price_deviation < -ma_deviation_threshold) OR
        # (lagged_correlation < 0 AND leading_asset_price_change < -significant_movement_threshold AND lagging_asset_price_deviation < -ma_deviation_threshold)
        
        buy_positive_corr_case = (corr > 0 and 
                                  leading_asset_price_change > significant_movement_threshold and 
                                  lagging_asset_price_deviation < -MA_DEVIATION_THRESHOLD)
        
        buy_negative_corr_case = (corr < 0 and 
                                  leading_asset_price_change < -significant_movement_threshold and 
                                  lagging_asset_price_deviation < -MA_DEVIATION_THRESHOLD)
        
        if buy_positive_corr_case or buy_negative_corr_case:
            seen_targets.add(b)
            signals.append(BuySignal(pair=b, timestamp=ts, price=price, confidence=abs(corr)))
        
        # 7. Generate Sell Signal
        # (lagged_correlation > 0 AND leading_asset_price_change < -significant_movement_threshold AND lagging_asset_price_deviation > ma_deviation_threshold) OR
        # (lagged_correlation < 0 AND leading_asset_price_change > significant_movement_threshold AND lagging_asset_price_deviation > ma_deviation_threshold)
        
        sell_positive_corr_case = (corr > 0 and 
                                   leading_asset_price_change < -significant_movement_threshold and 
                                   lagging_asset_price_deviation > MA_DEVIATION_THRESHOLD)
        
        sell_negative_corr_case = (corr < 0 and 
                                   leading_asset_price_change > significant_movement_threshold and 
                                   lagging_asset_price_deviation > MA_DEVIATION_THRESHOLD)
        
        if sell_positive_corr_case or sell_negative_corr_case:
            seen_targets.add(b)
            signals.append(SellSignal(pair=b, timestamp=ts, price=price, confidence=abs(corr)))

    return signals