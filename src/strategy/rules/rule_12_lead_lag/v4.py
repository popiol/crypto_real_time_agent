from __future__ import annotations

import math
import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# ── Constants for the original rule ───────────────────────────────────────────
MIN_CANDLES = 20  # minimum warm candles per asset for reliable correlation
MAX_LAG = 3  # maximum lead time in hours to consider
CORR_THRESHOLD = 0.5  # minimum Pearson r to treat a lag as a real relationship
LEAD_THRESHOLD = 0.01  # A's k-hour cumulative return must exceed this to signal

# ── Constants for the enhanced rule's filters ─────────────────────────────────
VWMA_PERIOD = 20  # Period for Volume-Weighted Moving Average
ATR_PERIOD = 14  # Period for Average True Range
AVERAGE_ATR_PERIOD = 50  # Period for SMA of ATR for dynamic threshold
VOLATILITY_THRESHOLD_FACTOR = 1.5  # Multiplier for average true range


# ── Pair cache (refreshed when warm tier changes, i.e. ~hourly) ───────────────
_cached_pairs: list[tuple[str, str, int, float]] = []  # (A, B, lag, corr)
_cache_key: str = ""


# ── Helper functions for base lead-lag logic ──────────────────────────────────

def _returns(closes: list[float]) -> np.ndarray:
    arr = np.array(closes, dtype=np.float64)
    return (arr[1:] - arr[:-1]) / np.where(arr[:-1] != 0, arr[:-1], 1.0)


def _best_lag_corr(r_a: np.ndarray, r_b: np.ndarray) -> tuple[int, float]:
    """Return (best_lag, corr) maximising corr(r_A[t], r_B[t+k]) over k."""
    n = min(len(r_a), len(r_b))
    best_lag, best_corr = 0, 0.0
    for k in range(1, MAX_LAG + 1):
        if n - k < 5:  # Need at least 5 points for correlation
            break
        c = float(np.corrcoef(r_a[: n - k], r_b[k:n])[0, 1])
        if math.isfinite(c) and c > best_corr:
            best_corr, best_lag = c, k
    return best_lag, best_corr


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

    # Removed logging statement as per requirements

    return _cached_pairs


# ── Helper functions for confirmation filters ─────────────────────────────────

def _calculate_vwma(prices: list[float], volumes_proxy: list[float], period: int) -> float | None:
    """
    Calculates the Volume-Weighted Moving Average (VWMA).
    Assumes `volumes_proxy` contains weights representing volume.
    """
    if len(prices) < period or len(volumes_proxy) < period:
        return None
    
    prices_window = np.array(prices[-period:])
    volumes_window = np.array(volumes_proxy[-period:])
    
    sum_volumes = np.sum(volumes_window)
    if sum_volumes == 0:
        return None # Avoid division by zero if no effective volume
    
    return float(np.sum(prices_window * volumes_window) / sum_volumes)


def _calculate_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float] | None:
    """
    Calculates Average True Range (ATR) using Wilder's smoothing method.
    Returns a list of ATR values.
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i-1]
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_ranges.append(max(tr1, tr2, tr3))

    if len(true_ranges) < period:
        return None

    atr_values = []
    # Initial ATR is the simple moving average of the first 'period' True Ranges
    initial_atr = np.mean(true_ranges[:period])
    atr_values.append(float(initial_atr))

    # Subsequent ATRs using Wilder's smoothing
    for i in range(period, len(true_ranges)):
        prev_atr = atr_values[-1]
        current_tr = true_ranges[i]
        next_atr = (prev_atr * (period - 1) + current_tr) / period
        atr_values.append(float(next_atr))
        
    return atr_values


def _calculate_sma(data_list: list[float], period: int) -> float | None:
    """Calculates the Simple Moving Average (SMA) of the last 'period' values."""
    if len(data_list) < period:
        return None
    return float(np.mean(data_list[-period:]))


# ── Main signal generation function ───────────────────────────────────────────

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    pairs = _get_pairs(data)
    if not pairs:
        return []

    signals: list[BuySignal | SellSignal] = []
    seen_targets: set[str] = set()

    # Minimum candles required for all filters:
    # VWMA_PERIOD (20)
    # ATR_PERIOD + AVERAGE_ATR_PERIOD (14 + 50 = 64)
    # So, min_warm_candles_for_filters = max(20, 64) = 64
    MIN_WARM_CANDLES_FOR_FILTERS = max(VWMA_PERIOD, ATR_PERIOD + AVERAGE_ATR_PERIOD)

    for a, b, lag, corr in pairs:
        if b in seen_targets:
            continue

        pd_a = data.get(a)
        pd_b = data.get(b)  # Lagging asset
        if pd_a is None or pd_b is None:
            continue
        # Ensure enough warm candles for lead-lag and for filters
        if len(pd_a.warm) < lag + 1 or len(pd_b.warm) < MIN_WARM_CANDLES_FOR_FILTERS or not pd_b.hot:
            continue

        # --- Base signal logic inherited from rule_12_lead_lag_v1 ---
        closes_a = [c.close for c in pd_a.warm]
        denom = closes_a[-lag - 1]
        if denom == 0:
            continue
        a_return = (closes_a[-1] - denom) / denom

        base_signal = None
        if a_return > LEAD_THRESHOLD:
            base_signal = 'BUY'
        elif a_return < -LEAD_THRESHOLD:
            base_signal = 'SELL'

        if base_signal is None:
            continue

        # --- Confirmation filters for lagging asset (b) ---
        lagging_warm_candles = pd_b.warm
        current_lagging_price = lagging_warm_candles[-1].close

        # Calculate VWMA for lagging asset using `avg_spread_rel` as an inverse proxy for volume.
        # A smaller spread typically implies higher liquidity/volume.
        lagging_closes_vwma = [c.close for c in lagging_warm_candles]
        lagging_volumes_proxy = []
        for c in lagging_warm_candles:
            if c.avg_spread_rel > 0:
                lagging_volumes_proxy.append(1.0 / c.avg_spread_rel)
            else:
                # Handle zero spread: assume very high liquidity/volume
                lagging_volumes_proxy.append(1e6) 

        vwma_val = _calculate_vwma(lagging_closes_vwma, lagging_volumes_proxy, VWMA_PERIOD)
        if vwma_val is None:
            continue

        # Calculate ATR for lagging asset
        lagging_highs = [c.high for c in lagging_warm_candles]
        lagging_lows = [c.low for c in lagging_warm_candles]
        lagging_closes_atr = [c.close for c in lagging_warm_candles]

        atr_values = _calculate_atr(lagging_highs, lagging_lows, lagging_closes_atr, ATR_PERIOD)
        if atr_values is None:
            continue
        current_atr = atr_values[-1]

        # Calculate historical average ATR for dynamic threshold
        average_atr_val = _calculate_sma(atr_values, AVERAGE_ATR_PERIOD)
        if average_atr_val is None:
            continue
        dynamic_volatility_threshold = average_atr_val * VOLATILITY_THRESHOLD_FACTOR

        # --- Apply confirmation logic ---
        confirmation_met = False
        if current_atr > dynamic_volatility_threshold:
            if base_signal == 'BUY' and current_lagging_price > vwma_val:
                confirmation_met = True
            elif base_signal == 'SELL' and current_lagging_price < vwma_val:
                confirmation_met = True

        # If all conditions are met, emit the signal
        if confirmation_met:
            ts = pd_b.hot[-1].polled_at
            price = pd_b.hot[-1].last_price
            seen_targets.add(b)
            if base_signal == 'BUY':
                signals.append(BuySignal(pair=b, timestamp=ts, price=price, confidence=corr))
            else:  # base_signal == 'SELL'
                signals.append(SellSignal(pair=b, timestamp=ts, price=price, confidence=corr))

    return signals