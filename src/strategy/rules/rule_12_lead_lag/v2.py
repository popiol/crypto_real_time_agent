"""Rule XX — Enhanced Lead-Lag with Dynamic Correlation and Volume Confirmation."""

from __future__ import annotations
import math
import numpy as np
from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick
from datetime import datetime

# --- Parameters ---
# These parameters assume a specific leading and lagging asset for this rule instance.
# In a real system, these might be configured per rule deployment or discovered dynamically.
# For this implementation, we pick a fixed pair for demonstration.
# TODO: Replace with actual asset symbols relevant to your trading environment.
LEADING_ASSET = "ETH-USD"  # Example: Ethereum
LAGGING_ASSET = "BTC-USD"  # Example: Bitcoin

LAG_PERIOD = 1  # The time lag for the lagging asset to react, in warm candles (hours).
                # A value of 1 means leading asset's movement 1 hour ago.
CORRELATION_WINDOW = 24  # Window for rolling correlation calculation, in warm candles (hours).
MIN_ABSOLUTE_CORRELATION_THRESHOLD = 0.6  # Minimum absolute Pearson correlation coefficient.
VOLUME_MA_WINDOW = 60  # Window for average volume calculation, in hot ticks (seconds).
                        # Using volume_24h from Tick objects.
VOLUME_MULTIPLIER = 1.5  # Multiplier for above-average volume, e.g., 1.5.
LEADING_PRICE_CHANGE_THRESHOLD = 0.005  # Minimum percentage change in leading asset to trigger signal, e.g., 0.005 (0.5%).

# Minimum data points required for calculations
# For correlation: need enough candles for returns, then for lagging, then for window.
# (N candles -> N-1 returns). If LAG_PERIOD=1, then leading_returns[:-1] and lagging_returns[1:].
# The shorter of these series must have at least CORRELATION_WINDOW elements.
# This means original prices must have at least CORRELATION_WINDOW + LAG_PERIOD + 1 for leading
# and CORRELATION_WINDOW + LAG_PERIOD + 1 for lagging if LAG_PERIOD > 0.
# If LAG_PERIOD=0, then CORRELATION_WINDOW + 1.
# A safe general check: max(CORRELATION_WINDOW + LAG_PERIOD, LAG_PERIOD + 1) + 1
MIN_CANDLES_FOR_CORRELATION = CORRELATION_WINDOW + LAG_PERIOD + 1
MIN_CANDLES_FOR_LEADING_CHANGE = LAG_PERIOD + 2 # Needs prices at [-LAG_PERIOD-1] and [-LAG_PERIOD-2]
MIN_TICKS_FOR_VOLUME = VOLUME_MA_WINDOW


# --- Helper Functions ---

def _calculate_returns(prices: list[float]) -> np.ndarray:
    """Computes percentage returns from a list of close prices."""
    arr = np.array(prices, dtype=np.float64)
    # Handle division by zero for prices that might be zero or near zero
    return (arr[1:] - arr[:-1]) / np.where(arr[:-1] != 0, arr[:-1], 1e-9)

def _calculate_lagged_correlation(
    leading_prices: list[float],
    lagging_prices: list[float],
    lag_period: int,
    window: int
) -> float | None:
    """
    Computes the rolling correlation between lagged leading asset returns and
    current lagging asset returns over a specified window.
    Returns the most recent correlation value.
    """
    # Ensure enough raw price data to calculate the required returns series
    if len(leading_prices) < lag_period + 2 or len(lagging_prices) < lag_period + 2:
        return None

    leading_returns = _calculate_returns(leading_prices)
    lagging_returns = _calculate_returns(lagging_prices)

    # Align the return series for lagged correlation:
    # We want to correlate (leading_return at t - lag_period) with (lagging_return at t)
    # If lag_period is 1, we correlate r_L[t-1] with r_G[t].
    # In 0-indexed arrays, r_L[i] is the return for the period ending at candle i+1.
    # So, `r_L[idx]` corresponds to candle `idx+1`.
    # `r_L[idx_L]` (return at candle `idx_L+1`) should correlate with `r_G[idx_L + lag_period]`
    # (return at candle `idx_L + lag_period + 1`).

    # `leading_returns_aligned` will contain `r_L[0], r_L[1], ..., r_L[N - 1 - lag_period]`
    # `lagging_returns_aligned` will contain `r_G[lag_period], r_G[lag_period+1], ..., r_G[N - 1]`
    # where N is the length of the original *return* series.

    if lag_period >= len(leading_returns) or lag_period >= len(lagging_returns):
        return None # Not enough returns to create lagged series

    leading_returns_aligned = leading_returns[:-lag_period] if lag_period > 0 else leading_returns
    lagging_returns_aligned = lagging_returns[lag_period:]

    # Take the minimum length of the two aligned series
    min_series_len = min(len(leading_returns_aligned), len(lagging_returns_aligned))

    if min_series_len < window:
        return None # Not enough aligned data points for the correlation window

    # Take the most recent 'window' pairs for correlation
    r_L_final = leading_returns_aligned[min_series_len - window : min_series_len]
    r_G_final = lagging_returns_aligned[min_series_len - window : min_series_len]

    # Need at least 2 data points for correlation calculation
    if len(r_L_final) < 2 or len(r_G_final) < 2:
        return None

    try:
        corr = float(np.corrcoef(r_L_final, r_G_final)[0, 1])
        if math.isfinite(corr):
            return corr
    except Exception: # Catch potential errors like all returns being zero (std dev = 0)
        pass
    return None

def _calculate_rolling_average(values: list[float], window: int) -> float | None:
    """Calculates the average of the last 'window' values."""
    if len(values) < window:
        return None
    # .item() to convert numpy scalar to Python float
    return np.mean(values[-window:]).item()


# --- Signal Generation ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    pd_leading = data.get(LEADING_ASSET)
    pd_lagging = data.get(LAGGING_ASSET)

    if pd_leading is None or pd_lagging is None:
        return []

    # --- 1. Extract Price Histories (Warm Candles) ---
    leading_prices = [c.close for c in pd_leading.warm]
    lagging_prices = [c.close for c in pd_lagging.warm]

    # --- 2. Extract Volume Histories (Hot Ticks) ---
    # WarmCandle does not contain volume. Using volume_24h from hot ticks.
    # This means VOLUME_MA_WINDOW refers to number of ticks, not candles.
    lagging_volumes_24h = [t.volume_24h for t in pd_lagging.hot]

    # --- Data Sufficiency Checks ---
    if len(leading_prices) < MIN_CANDLES_FOR_CORRELATION or \
       len(lagging_prices) < MIN_CANDLES_FOR_CORRELATION:
        return [] # Not enough historical candles for correlation calculation

    if len(leading_prices) < MIN_CANDLES_FOR_LEADING_CHANGE:
        return [] # Not enough candles to determine leading asset's trigger change

    if len(lagging_volumes_24h) < MIN_TICKS_FOR_VOLUME:
        return [] # Not enough recent ticks for volume confirmation

    # --- 3. Calculate Dynamic Lagged Correlation ---
    current_correlation = _calculate_lagged_correlation(
        leading_prices,
        lagging_prices,
        LAG_PERIOD,
        CORRELATION_WINDOW
    )

    if current_correlation is None or abs(current_correlation) < MIN_ABSOLUTE_CORRELATION_THRESHOLD:
        return [] # Correlation not strong enough or not calculable

    # --- 4. Calculate Leading Asset Trigger Change (LAG_PERIOD bars ago) ---
    # The pseudocode implies: (price at t-LAG_PERIOD-1) minus (price at t-LAG_PERIOD-2).
    # In 0-indexed list, prices are `[p_0, ..., p_{N-1}]`.
    # `p_{N-1}` is the most recent price.
    # `p_{N-1-LAG_PERIOD}` is the price LAG_PERIOD bars ago.
    # `p_{N-1-LAG_PERIOD-1}` is the price LAG_PERIOD+1 bars ago.
    # So we need `leading_prices[-LAG_PERIOD-1]` and `leading_prices[-LAG_PERIOD-2]`.
    
    price_at_lag_period_ago = leading_prices[-LAG_PERIOD-1]
    price_at_lag_period_plus_one_ago = leading_prices[-LAG_PERIOD-2]

    if price_at_lag_period_plus_one_ago == 0:
        return [] # Avoid division by zero

    leading_asset_trigger_change = (
        price_at_lag_period_ago - price_at_lag_period_plus_one_ago
    ) / price_at_lag_period_plus_one_ago

    # --- 5. Calculate Average Volume for Lagging Asset ---
    avg_volume_lagging = _calculate_rolling_average(lagging_volumes_24h, VOLUME_MA_WINDOW)
    current_volume_lagging = lagging_volumes_24h[-1]

    if avg_volume_lagging is None or avg_volume_lagging == 0:
        return [] # Cannot determine volume confirmation or average volume is zero

    # --- 6. Signal Generation Logic ---
    if current_volume_lagging > (avg_volume_lagging * VOLUME_MULTIPLIER):
        # Determine current timestamp and price for the signal from the latest tick
        if not pd_lagging.hot:
            return [] # No hot data for signal timestamp/price

        ts = pd_lagging.hot[-1].polled_at
        price = pd_lagging.hot[-1].last_price

        if price == 0: # Ensure we have a valid price
            return []

        if current_correlation > 0: # Positive correlation: leading up -> lagging up; leading down -> lagging down
            if leading_asset_trigger_change > LEADING_PRICE_CHANGE_THRESHOLD:
                signals.append(BuySignal(
                    pair=LAGGING_ASSET,
                    timestamp=ts,
                    price=price,
                    confidence=abs(current_correlation)
                ))
            elif leading_asset_trigger_change < -LEADING_PRICE_CHANGE_THRESHOLD:
                signals.append(SellSignal(
                    pair=LAGGING_ASSET,
                    timestamp=ts,
                    price=price,
                    confidence=abs(current_correlation)
                ))
        elif current_correlation < 0: # Negative correlation: leading up -> lagging down; leading down -> lagging up
            if leading_asset_trigger_change < -LEADING_PRICE_CHANGE_THRESHOLD: # Leading asset moved down
                signals.append(BuySignal( # Due to negative correlation, lagging asset expected to go up
                    pair=LAGGING_ASSET,
                    timestamp=ts,
                    price=price,
                    confidence=abs(current_correlation)
                ))
            elif leading_asset_trigger_change > LEADING_PRICE_CHANGE_THRESHOLD: # Leading asset moved up
                signals.append(SellSignal( # Due to negative correlation, lagging asset expected to go down
                    pair=LAGGING_ASSET,
                    timestamp=ts,
                    price=price,
                    confidence=abs(current_correlation)
                ))

    return signals