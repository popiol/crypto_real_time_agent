from __future__ import annotations

import math
import numpy as np
import logging

# Assuming these models are available from src.agent.models in the execution environment.
# For a self-contained module, we include dummy definitions.
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# Dummy model definitions to make the module self-contained for testing/linting.
# These should be replaced by actual imports from src.agent.models in a real deployment.
class Tick(BaseModel):
    pair: str
    polled_at: datetime
    last_price: float
    bid_price: float
    bid_volume: float
    ask_price: float
    ask_volume: float
    volume_24h: float = 0.0
    mid_price: float
    spread_abs: float
    spread_rel: float
    order_book: Optional[object] = None # Simplified, actual type is OrderBook

class WarmCandle(BaseModel):
    hour: datetime
    open_price: float
    high: float
    low: float
    close: float
    avg_spread_rel: float = 0.0

class ColdMonth(BaseModel):
    month: str
    min_price: float
    max_price: float
    avg_price: float
    avg_daily_spread: float
    candle_count: int
    last_candle_hour: datetime

class PairData(BaseModel):
    hot: List[Tick] = Field(default=[])
    warm: List[WarmCandle] = Field(default=[])
    cold: List[ColdMonth] = Field(default=[])

class BuySignal(BaseModel):
    pair: str
    timestamp: datetime
    price: float
    rule_id: str = ""
    confidence: Optional[float] = None

class SellSignal(BaseModel):
    pair: str
    timestamp: datetime
    price: float
    rule_id: str = ""
    confidence: Optional[float] = None

MarketData = Dict[str, PairData]
# End dummy model definitions


MIN_CANDLES = 20  # minimum warm candles per asset for reliable correlation
MAX_LAG = 3  # maximum lead time in hours to consider
CORR_THRESHOLD = 0.5  # minimum Pearson r to treat a lag as a real relationship

# New parameters for "Lead-Lag with Lagging Asset Bollinger Band Confirmation"
LEADING_LOOKBACK = 5  # Number of additional hours to look back for leading asset return, beyond the lag period
LEADING_BUY_THRESHOLD = 0.01  # Leading asset's return must exceed this for a Buy signal
LEADING_SELL_THRESHOLD = -0.01 # Leading asset's return must fall below this for a Sell signal
BB_PERIOD = 20  # Period for Bollinger Bands calculation on the lagging asset (e.g., 20 hours)
BB_STD_DEV = 2.0  # Standard deviation multiplier for Bollinger Bands (e.g., 2.0 for 2-sigma bands)


# ── Pair cache (refreshed when warm tier changes, i.e. ~hourly) ───────────────
_cached_pairs: list[tuple[str, str, int, float]] = []  # (A, B, lag, corr)
_cache_key: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _returns(closes: list[float]) -> np.ndarray:
    """Calculates percentage returns from a list of close prices."""
    arr = np.array(closes, dtype=np.float64)
    # Ensure no division by zero for returns calculation
    return (arr[1:] - arr[:-1]) / np.where(arr[:-1] != 0, arr[:-1], 1.0)


def _best_lag_corr(r_a: np.ndarray, r_b: np.ndarray) -> tuple[int, float]:
    """
    Return (best_lag, corr) maximising corr(r_A[t], r_B[t+k]) over k.
    Considers lags from 1 to MAX_LAG.
    """
    n = min(len(r_a), len(r_b))
    best_lag, best_corr = 0, 0.0
    for k in range(1, MAX_LAG + 1):
        # Need at least 5 data points for meaningful correlation after applying lag
        if n - k < 5:
            break
        
        # Ensure enough data points for correlation calculation (at least 2)
        if len(r_a[: n - k]) < 2 or len(r_b[k:n]) < 2:
            continue
        
        # Check for constant arrays, which lead to NaN correlation
        if np.std(r_a[: n - k]) == 0 or np.std(r_b[k:n]) == 0:
            continue

        c = float(np.corrcoef(r_a[: n - k], r_b[k:n])[0, 1])
        if math.isfinite(c) and c > best_corr:
            best_corr, best_lag = c, k
    return best_lag, best_corr


# ── Pair detection (cached) ───────────────────────────────────────────────────


def _detect_pairs(
    asset_returns: dict[str, np.ndarray],
) -> list[tuple[str, str, int, float]]:
    """
    Detects leading-lagging asset pairs based on cross-correlation of returns.
    """
    pairs: list[tuple[str, str, int, float]] = []
    assets = list(asset_returns.keys())
    for a in assets:
        for b in assets:
            if b == a:  # An asset cannot lead itself
                continue
            lag, corr = _best_lag_corr(asset_returns[a], asset_returns[b])
            if corr > CORR_THRESHOLD:
                pairs.append((a, b, lag, corr))
    return pairs


def _get_pairs(data: MarketData) -> list[tuple[str, str, int, float]]:
    """
    Retrieves or detects leading-lagging pairs. Results are cached hourly.
    """
    global _cached_pairs, _cache_key

    # Create a cache key based on the latest warm candle hour for each pair.
    # This ensures the cache is refreshed when underlying hourly data changes.
    key_parts = []
    for pair, pd in sorted(data.items()):
        if pd.warm:
            key_parts.append(f"{pair}:{pd.warm[-1].hour.isoformat()}")
    
    key = "|".join(key_parts)

    # Return cached pairs if the cache key matches and is not empty
    if key and key == _cache_key:
        return _cached_pairs

    asset_returns: dict[str, np.ndarray] = {}
    for pair, pd in data.items():
        if len(pd.warm) >= MIN_CANDLES:
            asset_returns[pair] = _returns([c.close for c in pd.warm])

    _cached_pairs = _detect_pairs(asset_returns)
    _cache_key = key  # Update cache key even if no pairs were found

    if _cached_pairs:
        logger_pairs = [(a, b, k, round(c, 2)) for a, b, k, c in _cached_pairs]
        # Using a logger. In a real system, this logger would be configured.
        logging.getLogger(__name__).debug("Lead-lag pairs detected: %s", logger_pairs)

    return _cached_pairs


# ── Signal generation ─────────────────────────────────────────────────────────


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on a lead-lag relationship confirmed by
    Bollinger Bands on the lagging asset.
    """
    pairs = _get_pairs(data)
    if not pairs:
        return []

    signals: list[BuySignal | SellSignal] = []
    seen_targets: set[str] = set()

    for a, b, lag, corr in pairs:
        # Ensure only one signal is emitted per target asset per cycle
        if b in seen_targets:
            continue

        pd_a = data.get(a)
        pd_b = data.get(b)
        if pd_a is None or pd_b is None:
            continue

        # --- Data sufficiency checks ---
        # 1. For leading asset (A) return calculation:
        #    We need the current close (index -1) and the close from `LEADING_LOOKBACK + lag` hours ago.
        #    This requires a total of `LEADING_LOOKBACK + lag + 1` warm candles.
        required_candles_a = LEADING_LOOKBACK + lag + 1
        if len(pd_a.warm) < required_candles_a:
            continue

        # 2. For lagging asset (B) Bollinger Bands calculation:
        #    We need at least `BB_PERIOD` warm candles to calculate the moving average and standard deviation.
        if len(pd_b.warm) < BB_PERIOD:
            continue

        # 3. For current lagging price: need at least one hot tick.
        if not pd_b.hot:
            continue

        # --- Calculate leading asset (A) return ---
        closes_a = [c.close for c in pd_a.warm]
        
        # The pseudocode implies: (current_close - close_at_lookback) / close_at_lookback
        # `close_at_lookback` is `LEADING_LOOKBACK + lag` hours prior to the current warm candle's close.
        # In Python list indexing, `closes_a[-1]` is the current close.
        # `closes_a[-(LEADING_LOOKBACK + lag + 1)]` gives the close from `LEADING_LOOKBACK + lag` periods ago.
        start_price_idx = -(LEADING_LOOKBACK + lag + 1)
        
        leading_start_price = closes_a[start_price_idx]
        leading_current_price = closes_a[-1]

        if leading_start_price == 0:  # Avoid division by zero
            continue
        leading_asset_return = (leading_current_price - leading_start_price) / leading_start_price

        # --- Calculate Bollinger Bands for lagging asset (B) ---
        closes_b = [c.close for c in pd_b.warm]
        
        # Extract the last `BB_PERIOD` close prices for Bollinger Band calculation.
        bb_closes = np.array(closes_b[-BB_PERIOD:]) 

        lagging_ma = np.mean(bb_closes)
        lagging_std = np.std(bb_closes)

        # Calculate upper and lower Bollinger Bands
        upper_band = lagging_ma + (lagging_std * BB_STD_DEV)
        lower_band = lagging_ma - (lagging_std * BB_STD_DEV)

        # --- Get current lagging price and timestamp from the hot (real-time) data ---
        current_lagging_price = pd_b.hot[-1].last_price
        ts = pd_b.hot[-1].polled_at

        # --- Generate signals based on combined conditions ---
        # Buy signal condition:
        # 1. Leading asset's return exceeds a positive threshold (predicting upward movement).
        # 2. Lagging asset's current price is below its lower Bollinger Band (indicating an oversold condition).
        if leading_asset_return > LEADING_BUY_THRESHOLD and current_lagging_price < lower_band:
            seen_targets.add(b)
            signals.append(BuySignal(pair=b, timestamp=ts, price=current_lagging_price, confidence=corr))
        
        # Sell signal condition:
        # 1. Leading asset's return falls below a negative threshold (predicting downward movement).
        # 2. Lagging asset's current price is above its upper Bollinger Band (indicating an overbought condition).
        elif leading_asset_return < LEADING_SELL_THRESHOLD and current_lagging_price > upper_band:
            seen_targets.add(b)
            signals.append(SellSignal(pair=b, timestamp=ts, price=current_lagging_price, confidence=corr))

    return signals