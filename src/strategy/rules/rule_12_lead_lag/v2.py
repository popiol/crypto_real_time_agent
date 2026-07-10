from __future__ import annotations

import math
import numpy as np
from collections import deque
import logging
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle

# Constants from original rule
MIN_CANDLES = 20  # minimum warm candles per asset for reliable correlation
MAX_LAG = 3  # maximum lead time in hours to consider
CORR_THRESHOLD = 0.5  # minimum Pearson r (absolute) to treat a lag as a real relationship
LEAD_THRESHOLD = 0.01  # A's k-hour cumulative return must exceed this to signal

# New constants for refinement
MIN_CORRELATION_PERCENTILE = 0.75  # e.g., require absolute correlation to be in top 25% of its history
HISTORICAL_CORR_WINDOW = 24 * 7 # Store last 7 days (24*7 hourly updates) of correlations for percentile calculation
LAGGING_ASSET_CONFIRMATION_LOOKBACK = 2  # periods (ticks) for lagging asset's initial movement confirmation
LAGGING_ASSET_CONFIRMATION_THRESHOLD_FACTOR = 0.1  # e.g., 0.1 * ATR for initial movement
ATR_PERIOD = 14  # Standard ATR period for warm candles

# ── Pair cache (refreshed when warm tier changes, i.e. ~hourly) ───────────────
_cached_pairs: list[tuple[str, str, int, float]] = []  # (A, B, lag, corr)
_cache_key: str = ""

# ── Historical correlations cache ─────────────────────────────────────────────
# Store a deque of absolute correlation values for each (A, B, lag) combination
_historical_abs_correlations: dict[str, deque[float]] = {}

# ── Helpers ───────────────────────────────────────────────────────────────────

_logger = logging.getLogger(__name__)

def _returns(closes: list[float]) -> np.ndarray:
    """Calculates period-over-period returns from a list of close prices."""
    arr = np.array(closes, dtype=np.float64)
    # Ensure no division by zero; replace zero denominators with 1 to avoid NaN/inf in returns
    # This makes returns 0 if previous close was 0, which is reasonable for price data.
    return (arr[1:] - arr[:-1]) / np.where(arr[:-1] != 0, arr[:-1], 1.0)


def _best_lag_corr(r_a: np.ndarray, r_b: np.ndarray) -> tuple[int, float]:
    """
    Return (best_lag, actual_corr) maximising abs(corr(r_A[t], r_B[t+k])) over k.
    This allows detecting strong negative lead-lag relationships too.
    """
    n = min(len(r_a), len(r_b))
    best_lag, best_abs_corr, actual_corr_at_best_lag = 0, 0.0, 0.0
    for k in range(1, MAX_LAG + 1):
        # np.corrcoef needs at least 2 points for calculation, 5 is a safer minimum.
        if n - k < 5:
            break # If n-k is too small, subsequent k's will also be too small.
        
        # Ensure there's enough data for both series after lagging
        if len(r_a[:n-k]) < 2 or len(r_b[k:n]) < 2:
            continue

        c = float(np.corrcoef(r_a[: n - k], r_b[k:n])[0, 1])
        if math.isfinite(c) and abs(c) > best_abs_corr:
            best_abs_corr, best_lag, actual_corr_at_best_lag = abs(c), k, c
    return best_lag, actual_corr_at_best_lag


def _calculate_atr(warm_candles: list[WarmCandle], period: int) -> float:
    """Calculates Average True Range (ATR) from a list of WarmCandle objects."""
    if len(warm_candles) < period + 1:
        return 0.0

    highs = [c.high for c in warm_candles]
    lows = [c.low for c in warm_candles]
    closes = [c.close for c in warm_candles]

    trs = []
    for i in range(1, len(closes)):
        h = highs[i]
        l = lows[i]
        prev_close = closes[i-1]
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
    
    if len(trs) < period: # Not enough TRs to calculate initial ATR
        return 0.0

    # Initial ATR (Simple Moving Average of first 'period' True Ranges)
    atr_vals = [sum(trs[:period]) / period]
    
    # Subsequent ATR using Wilder's smoothing method
    for i in range(period, len(trs)):
        atr_val = (atr_vals[-1] * (period - 1) + trs[i]) / period
        atr_vals.append(atr_val)
            
    return atr_vals[-1] if atr_vals else 0.0


# ── Pair detection (cached) ───────────────────────────────────────────────────


def _detect_pairs(
    asset_returns: dict[str, np.ndarray],
    data: MarketData # Added to access full PairData for ATR calculation
) -> list[tuple[str, str, int, float]]:
    """
    Detects lead-lag pairs, updating historical absolute correlations.
    """
    pairs: list[tuple[str, str, int, float]] = []
    assets = list(asset_returns.keys())
    
    # Create a set of all current (A, B, lag) combinations that are being evaluated
    current_evaluated_keys = set()

    for i, a_pair_name in enumerate(assets):
        for b_pair_name in assets:
            if b_pair_name == a_pair_name:
                continue
            
            # Ensure enough warm candles for ATR calculation later for the lagging asset
            if len(data.get(b_pair_name, PairData()).warm) < ATR_PERIOD + 1:
                _logger.debug(f"Insufficient warm data for ATR for {b_pair_name}. Skipping pair detection for {a_pair_name}-{b_pair_name}.")
                continue

            lag, corr = _best_lag_corr(asset_returns[a_pair_name], asset_returns[b_pair_name])
            
            # Only proceed if a valid lag and correlation (non-zero abs_corr) was found
            if lag == 0 and corr == 0.0:
                continue

            # Update historical correlations for this specific (A, B, lag)
            corr_key = f"{a_pair_name}_{b_pair_name}_{lag}"
            if corr_key not in _historical_abs_correlations:
                _historical_abs_correlations[corr_key] = deque(maxlen=HISTORICAL_CORR_WINDOW)
            _historical_abs_correlations[corr_key].append(abs(corr)) # Store absolute correlation
            current_evaluated_keys.add(corr_key)

            # Initial filter: absolute correlation must exceed a fixed threshold
            if abs(corr) > CORR_THRESHOLD:
                pairs.append((a_pair_name, b_pair_name, lag, corr))
    
    # Prune historical correlations for pairs that are no longer being evaluated
    # This prevents memory growth for dead pairs and ensures percentile is based on relevant history
    keys_to_remove = [k for k in _historical_abs_correlations if k not in current_evaluated_keys]
    for k in keys_to_remove:
        del _historical_abs_correlations[k]
        _logger.debug(f"Pruned historical correlation for {k} as it's no longer being evaluated.")

    return pairs


def _get_pairs(data: MarketData) -> list[tuple[str, str, int, float]]:
    global _cached_pairs, _cache_key

    # Cache key: latest warm candle hour for each pair (changes once per hour)
    key_parts = []
    for pair, pd in sorted(data.items()):
        if pd.warm:
            # Check if 'hour' attribute exists and is a datetime object
            if hasattr(pd.warm[-1], 'hour') and isinstance(pd.warm[-1].hour, datetime):
                key_parts.append(f"{pair}:{pd.warm[-1].hour.isoformat()}")
            else:
                # Fallback if 'hour' is not a datetime object (e.g., just a string or int)
                key_parts.append(f"{pair}:{pd.warm[-1].hour}")
    key = "|".join(key_parts)

    # Return cached pairs if data hasn't changed and cache is not empty
    if key == _cache_key and _cached_pairs:
        return _cached_pairs

    asset_returns: dict[str, np.ndarray] = {}
    for pair, pd in data.items():
        # Need enough warm candles for both return calculation and ATR for lagging asset checks
        if len(pd.warm) >= MIN_CANDLES:
            asset_returns[pair] = _returns([c.close for c in pd.warm])
        else:
            _logger.debug(f"Insufficient warm data for {pair} (have {len(pd.warm)}, need {MIN_CANDLES}). Skipping return calculation.")


    # Pass full data to _detect_pairs for ATR data access and other checks
    _cached_pairs = _detect_pairs(asset_returns, data)
    _cache_key = key

    if _cached_pairs:
        logger_pairs = [(a, b, k, round(c, 2)) for a, b, k, c in _cached_pairs]
        _logger.debug("Lead-lag pairs detected: %s", logger_pairs)
    else:
        _logger.debug("No lead-lag pairs detected.")

    return _cached_pairs


# ── Signal generation ─────────────────────────────────────────────────────────

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on a refined lead-lag relationship rule.
    """
    signals: list[BuySignal | SellSignal] = []
    
    if not data:
        _logger.debug("No market data provided. Returning no signals.")
        return []

    pairs = _get_pairs(data)
    if not pairs:
        return []

    seen_targets: set[str] = set() # To ensure at most one signal per target asset per cycle

    for a, b, lag, corr in pairs:
        if b in seen_targets:
            continue

        pd_a = data.get(a)
        pd_b = data.get(b)
        if pd_a is None or pd_b is None:
            _logger.warning(f"Missing PairData for asset {a} or {b}. Skipping pair.")
            continue
        
        # Ensure enough warm data for leader's return calculation
        if len(pd_a.warm) < lag + 1:
            _logger.debug(f"Insufficient warm data for leader {a} (need {lag+1}, have {len(pd_a.warm)}). Skipping pair.")
            continue
        
        # Ensure enough warm data for lagging asset's ATR calculation
        if len(pd_b.warm) < ATR_PERIOD + 1:
            _logger.debug(f"Insufficient warm data for lagging asset {b} for ATR (need {ATR_PERIOD+1}, have {len(pd_b.warm)}). Skipping pair.")
            continue

        # Ensure enough hot data for lagging asset's confirmation movement
        if len(pd_b.hot) < LAGGING_ASSET_CONFIRMATION_LOOKBACK + 1:
            _logger.debug(f"Insufficient hot data for lagging asset {b} for confirmation (need {LAGGING_ASSET_CONFIRMATION_LOOKBACK+1}, have {len(pd_b.hot)}). Skipping pair.")
            continue

        # 1. Dynamic Minimum Absolute Correlation Threshold
        corr_key = f"{a}_{b}_{lag}"
        historical_abs_corrs = _historical_abs_correlations.get(corr_key)
        
        # Need enough historical data points to calculate a meaningful percentile
        if not historical_abs_corrs or len(historical_abs_corrs) < MIN_CANDLES: # Using MIN_CANDLES as a heuristic for min history length
            _logger.debug(f"Insufficient historical correlations for {corr_key} (have {len(historical_abs_corrs)}). Skipping pair.")
            continue
        
        # Calculate dynamic threshold: require absolute correlation to be in top PERCENTILE of its history
        dynamic_min_abs_correlation = np.percentile(list(historical_abs_corrs), MIN_CORRELATION_PERCENTILE * 100)
        
        if abs(corr) < dynamic_min_abs_correlation:
            _logger.debug(f"Correlation for {corr_key} (abs={abs(corr):.2f}) below dynamic threshold ({dynamic_min_abs_correlation:.2f}). Skipping.")
            continue # Correlation is not strong enough dynamically

        # 2. Lagging Asset Confirmation
        current_atr = _calculate_atr(pd_b.warm, ATR_PERIOD)
        if current_atr <= 0: # ATR cannot be zero or negative.
            _logger.debug(f"ATR for {b} is zero or negative ({current_atr:.4f}). Skipping pair.")
            continue

        hot_prices_b = [t.last_price for t in pd_b.hot]
        
        # Calculate recent movement of the lagging asset over the confirmation lookback period
        lagging_recent_movement = hot_prices_b[-1] - hot_prices_b[-1 - LAGGING_ASSET_CONFIRMATION_LOOKBACK]
        lagging_confirmation_threshold = LAGGING_ASSET_CONFIRMATION_THRESHOLD_FACTOR * current_atr
        
        lagging_confirmation_buy = (lagging_recent_movement > 0) and (abs(lagging_recent_movement) > lagging_confirmation_threshold)
        lagging_confirmation_sell = (lagging_recent_movement < 0) and (abs(lagging_recent_movement) > lagging_confirmation_threshold)

        # 3. Leader's movement check (existing logic)
        closes_a = [c.close for c in pd_a.warm]
        denom = closes_a[-lag - 1]
        if denom == 0:
            _logger.debug(f"Denominator for leader {a} return calculation is zero. Skipping pair.")
            continue
        a_return = (closes_a[-1] - denom) / denom

        ts = pd_b.hot[-1].polled_at
        price = pd_b.hot[-1].last_price

        # Combine all conditions for signal generation
        # For a BUY signal: Leader predicts rise, positive correlation *above dynamic threshold*, and lagging asset shows initial upward confirmation.
        if a_return > LEAD_THRESHOLD and corr > dynamic_min_abs_correlation and lagging_confirmation_buy:
            seen_targets.add(b)
            signals.append(BuySignal(pair=b, timestamp=ts, price=price, confidence=abs(corr)))
            _logger.info(f"BUY Signal for {b} (Leader: {a}, Lag: {lag}, Corr: {corr:.2f}, DynCorrThreshold: {dynamic_min_abs_correlation:.2f}, A_Return: {a_return:.2f}, B_Move: {lagging_recent_movement:.4f}, ATR: {current_atr:.4f})")
        # For a SELL signal: Leader predicts fall, negative correlation *below negative dynamic threshold*, and lagging asset shows initial downward confirmation.
        elif a_return < -LEAD_THRESHOLD and corr < -dynamic_min_abs_correlation and lagging_confirmation_sell:
            seen_targets.add(b)
            signals.append(SellSignal(pair=b, timestamp=ts, price=price, confidence=abs(corr)))
            _logger.info(f"SELL Signal for {b} (Leader: {a}, Lag: {lag}, Corr: {corr:.2f}, DynCorrThreshold: {dynamic_min_abs_correlation:.2f}, A_Return: {a_return:.2f}, B_Move: {lagging_recent_movement:.4f}, ATR: {current_atr:.4f})")

    return signals