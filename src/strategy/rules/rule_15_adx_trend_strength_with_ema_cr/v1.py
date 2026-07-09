from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle
from datetime import datetime

# Rule parameters
ADX_PERIOD = 14
FAST_EMA_PERIOD = 9
SLOW_EMA_PERIOD = 20
ADX_THRESHOLD = 25.0

# Minimum candles required to calculate all indicators and check for crossovers.
# ADX needs at least 2 * ADX_PERIOD for the first non-NaN value
# (ADX_PERIOD for initial TR/DM smoothing, then another ADX_PERIOD for DX smoothing).
# EMAs need 'period' for the first non-NaN value.
# To check for a crossover, we need at least two valid values (current and previous).
MIN_CANDLES = max(FAST_EMA_PERIOD, SLOW_EMA_PERIOD, ADX_PERIOD * 2) + 1

# Unique identifier for this rule
RULE_ID = "070bf44f-bc32-40b9-890e-cb82fe795a95"


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates the Exponential Moving Average (EMA)."""
    if len(prices) < period:
        return np.full_like(prices, np.nan)

    ema = np.full_like(prices, np.nan)
    multiplier = 2 / (period + 1)

    # Initialize the first EMA with the simple moving average of the first 'period' values
    ema[period - 1] = np.mean(prices[:period])

    # Calculate subsequent EMA values
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema


def _wilder_smoothing(arr: np.ndarray, period: int) -> np.ndarray:
    """Applies Wilder's smoothing (RMA) to an array."""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    
    smoothed_arr = np.full_like(arr, np.nan)
    
    # First smoothed value is typically the simple average of the first 'period' values
    smoothed_arr[period - 1] = np.sum(arr[:period]) / period
    
    # Apply Wilder's smoothing for subsequent values
    for i in range(period, len(arr)):
        smoothed_arr[i] = (smoothed_arr[i - 1] * (period - 1) + arr[i]) / period
    return smoothed_arr


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Average Directional Index (ADX), Positive Directional Indicator (+DI),
    and Negative Directional Indicator (-DI).
    """
    if len(high) < MIN_CANDLES: # Check for overall data sufficiency based on the most demanding indicator
        return (
            np.full_like(close, np.nan), # ADX
            np.full_like(close, np.nan), # +DI
            np.full_like(close, np.nan), # -DI
        )

    # 1. Calculate Directional Movement (+DM, -DM) and True Range (TR)
    # These arrays will be len(prices) - 1, as they compare current to previous.
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.zeros_like(up_move)
    minus_dm = np.zeros_like(up_move)

    for i in range(len(up_move)):
        if up_move[i] > down_move[i] and up_move[i] > 0:
            plus_dm[i] = up_move[i]
        elif down_move[i] > up_move[i] and down_move[i] > 0:
            minus_dm[i] = down_move[i]

    tr = np.maximum(
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    )

    # 2. Wilder's Smoothing (RMA) for +DM, -DM, TR
    smoothed_plus_dm = _wilder_smoothing(plus_dm, period)
    smoothed_minus_dm = _wilder_smoothing(minus_dm, period)
    smoothed_tr = _wilder_smoothing(tr, period)

    # 3. Calculate +DI, -DI
    plus_di = np.full_like(smoothed_tr, np.nan)
    minus_di = np.full_like(smoothed_tr, np.nan)

    # Avoid division by zero and propagate NaNs
    valid_indices = np.where((smoothed_tr != 0) & (~np.isnan(smoothed_tr)))[0]
    plus_di[valid_indices] = (smoothed_plus_dm[valid_indices] / smoothed_tr[valid_indices]) * 100
    minus_di[valid_indices] = (smoothed_minus_dm[valid_indices] / smoothed_tr[valid_indices]) * 100
    
    # 4. Calculate DX
    dx = np.full_like(plus_di, np.nan)
    di_sum = plus_di + minus_di
    
    # Avoid division by zero and propagate NaNs
    valid_dx_indices = np.where((di_sum != 0) & (~np.isnan(di_sum)))[0]
    dx[valid_dx_indices] = (np.abs(plus_di[valid_dx_indices] - minus_di[valid_dx_indices]) / di_sum[valid_dx_indices]) * 100

    # 5. Calculate ADX (smoothed DX)
    adx = _wilder_smoothing(dx, period)

    # Pad the results with NaNs at the beginning to match the original `close` array length.
    # ADX, +DI, -DI are calculated from the second candle onwards, so they are `len(close) - 1` long.
    # Prepend one NaN to align them with the original candle data.
    adx_padded = np.full_like(close, np.nan)
    plus_di_padded = np.full_like(close, np.nan)
    minus_di_padded = np.full_like(close, np.nan)

    # Assign the calculated values, shifting by 1 to align with original `close` array
    # This also means the first `2*period - 1` elements will be NaN due to smoothing logic.
    if len(adx) > 0: # Ensure adx is not empty if MIN_CANDLES was not met, although outer check should prevent this.
        adx_padded[1:] = adx
        plus_di_padded[1:] = plus_di
        minus_di_padded[1:] = minus_di

    return adx_padded, plus_di_padded, minus_di_padded


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on ADX Trend Strength with EMA Crossover.
    Emits a Buy signal when ADX indicates a strong uptrend and a fast EMA crosses above a slow EMA.
    Emits a Sell signal when ADX indicates a strong downtrend and a fast EMA crosses below a slow EMA.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        
        # Ensure sufficient data for calculations
        if len(warm_candles) < MIN_CANDLES:
            continue

        # Extract required price data from WarmCandle objects
        high_prices = np.array([c.high for c in warm_candles])
        low_prices = np.array([c.low for c in warm_candles])
        close_prices = np.array([c.close for c in warm_candles])
        timestamps = [c.hour for c in warm_candles]

        # Calculate EMAs
        fast_ema = calculate_ema(close_prices, FAST_EMA_PERIOD)
        slow_ema = calculate_ema(close_prices, SLOW_EMA_PERIOD)

        # Calculate ADX, +DI, -DI
        adx_values, plus_di_values, minus_di_values = calculate_adx(high_prices, low_prices, close_prices, ADX_PERIOD)

        # Ensure we have valid (non-NaN) values for the latest and previous candles for all indicators
        if (
            np.isnan(fast_ema[-1]) or np.isnan(fast_ema[-2]) or
            np.isnan(slow_ema[-1]) or np.isnan(slow_ema[-2]) or
            np.isnan(adx_values[-1]) or np.isnan(plus_di_values[-1]) or np.isnan(minus_di_values[-1])
        ):
            continue

        # Get latest indicator values
        current_fast_ema = fast_ema[-1]
        previous_fast_ema = fast_ema[-2]
        current_slow_ema = slow_ema[-1]
        previous_slow_ema = slow_ema[-2]

        current_adx = adx_values[-1]
        current_plus_di = plus_di_values[-1]
        current_minus_di = minus_di_values[-1]

        current_price = close_prices[-1]
        current_timestamp = timestamps[-1]

        # Buy Signal conditions:
        # 1. ADX > ADX_THRESHOLD (strong trend)
        # 2. +DI > -DI (uptrend direction)
        # 3. Fast EMA crosses above Slow EMA (previous fast <= previous slow AND current fast > current slow)
        if (
            current_adx > ADX_THRESHOLD and
            current_plus_di > current_minus_di and
            previous_fast_ema <= previous_slow_ema and
            current_fast_ema > current_slow_ema
        ):
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=None, # Confidence is not specified, so set to None
                )
            )

        # Sell Signal conditions:
        # 1. ADX > ADX_THRESHOLD (strong trend)
        # 2. -DI > +DI (downtrend direction)
        # 3. Fast EMA crosses below Slow EMA (previous fast >= previous slow AND current fast < current slow)
        elif (
            current_adx > ADX_THRESHOLD and
            current_minus_di > current_plus_di and
            previous_fast_ema >= previous_slow_ema and
            current_fast_ema < current_slow_ema
        ):
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=None, # Confidence is not specified, so set to None
                )
            )

    return signals