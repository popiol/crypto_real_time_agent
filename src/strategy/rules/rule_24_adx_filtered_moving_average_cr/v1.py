from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule Parameters ---
RULE_ID = "573c4a89-8451-4c67-b04c-8e8016f1b533" # Idea ID
SHORT_MA_PERIOD = 10
LONG_MA_PERIOD = 30
ADX_PERIOD = 14
ADX_THRESHOLD = 25

# Minimum required candles for calculations
# ADX needs at least 2*ADX_PERIOD - 1 candles for its first valid value.
# EMAs need their respective periods.
# We need at least 2 data points for crossover check (current and previous).
MIN_CANDLES_REQUIRED = max(LONG_MA_PERIOD, (2 * ADX_PERIOD - 1)) + 1 # +1 for the last candle itself

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates Exponential Moving Average."""
    if len(prices) == 0:
        return np.array([])
    if len(prices) < period:
        return np.full_like(prices, np.nan)

    ema = np.zeros_like(prices)
    alpha = 2 / (period + 1)

    # Initialize the first EMA value with the first price
    ema[0] = prices[0]

    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_adx(high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates Average Directional Index (ADX)."""
    if not (len(high_prices) == len(low_prices) == len(close_prices)):
        raise ValueError("Price arrays must have the same length.")
    if len(high_prices) <= period: # Need enough data for initial TR/DM sum
        return np.full_like(high_prices, np.nan)

    adx_values_output = np.full_like(high_prices, np.nan)

    # Calculate True Range (TR)
    tr = np.zeros_like(high_prices)
    tr[0] = high_prices[0] - low_prices[0] # Initial TR, no previous close
    for i in range(1, len(high_prices)):
        tr[i] = max(
            high_prices[i] - low_prices[i],
            abs(high_prices[i] - close_prices[i-1]),
            abs(low_prices[i] - close_prices[i-1])
        )

    # Calculate Directional Movement (+DM, -DM)
    plus_dm = np.zeros_like(high_prices)
    minus_dm = np.zeros_like(high_prices)
    for i in range(1, len(high_prices)):
        up_move = high_prices[i] - high_prices[i-1]
        down_move = low_prices[i-1] - low_prices[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0

    # Smooth TR, +DM, -DM using Wilder's Smoothing
    # The first smoothed value is the sum of the first 'period' values.
    # Subsequent values use: smoothed_X[i] = smoothed_X[i-1] - (smoothed_X[i-1] / period) + current_X[i]
    
    smoothed_tr = np.zeros_like(high_prices)
    smoothed_plus_dm = np.zeros_like(high_prices)
    smoothed_minus_dm = np.zeros_like(high_prices)

    if len(high_prices) >= period:
        # Initial sum for the first 'period' values
        smoothed_tr[period-1] = np.sum(tr[0:period])
        smoothed_plus_dm[period-1] = np.sum(plus_dm[0:period])
        smoothed_minus_dm[period-1] = np.sum(minus_dm[0:period])

        # Apply Wilder's smoothing for subsequent values
        for i in range(period, len(high_prices)):
            smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1] / period) + tr[i]
            smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1] / period) + plus_dm[i]
            smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1] / period) + minus_dm[i]

    # Calculate +DI, -DI
    plus_di = np.zeros_like(high_prices)
    minus_di = np.zeros_like(high_prices)

    for i in range(period - 1, len(high_prices)):
        if smoothed_tr[i] != 0:
            plus_di[i] = 100 * (smoothed_plus_dm[i] / smoothed_tr[i])
            minus_di[i] = 100 * (smoothed_minus_dm[i] / smoothed_tr[i])
        # else DI is 0, which is correctly initialized

    # Calculate DX
    dx = np.zeros_like(high_prices)
    # DX starts from the point where DI values become valid (index period-1)
    for i in range(period - 1, len(high_prices)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * abs((plus_di[i] - minus_di[i]) / di_sum)
        # else DX is 0, correctly initialized

    # Calculate ADX (smoothed DX)
    # ADX smoothing typically starts after another 'period' values of DX are available.
    # The first valid ADX value will be at index (2 * period - 2).
    if len(high_prices) >= (2 * period - 1): # Need at least 2*period - 1 candles to get the first ADX value
        # Initial ADX is the average of the first 'period' DX values
        # These DX values start from index `period-1` and go up to `2*period-2`.
        first_adx_idx = 2 * period - 2
        adx_values_output[first_adx_idx] = np.mean(dx[period-1 : first_adx_idx + 1])

        # Apply Wilder's smoothing for subsequent values
        for i in range(first_adx_idx + 1, len(high_prices)):
            adx_values_output[i] = (adx_values_output[i-1] * (period - 1) + dx[i]) / period
    
    return adx_values_output

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm
        if len(candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract prices as numpy arrays
        high_prices = np.array([c.high for c in candles], dtype=np.float64)
        low_prices = np.array([c.low for c in candles], dtype=np.float64)
        close_prices = np.array([c.close for c in candles], dtype=np.float64)
        timestamps = [c.hour for c in candles]

        # Calculate EMAs
        short_ma = calculate_ema(close_prices, SHORT_MA_PERIOD)
        long_ma = calculate_ema(close_prices, LONG_MA_PERIOD)

        # Calculate ADX
        adx_values = calculate_adx(high_prices, low_prices, close_prices, ADX_PERIOD)
        
        # Ensure we have valid data points for the latest calculations and crossover checks.
        # We need the last two MA values and the last ADX value to be non-NaN.
        if (
            np.isnan(short_ma[-1]) or np.isnan(long_ma[-1]) or np.isnan(adx_values[-1]) or
            np.isnan(short_ma[-2]) or np.isnan(long_ma[-2])
        ):
            continue

        # Check for Buy Signal
        # Short MA crosses above Long MA AND ADX is above threshold
        if (short_ma[-2] <= long_ma[-2] and short_ma[-1] > long_ma[-1] and adx_values[-1] > ADX_THRESHOLD):
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamps[-1],
                price=close_prices[-1],
                rule_id=RULE_ID,
                confidence=min(1.0, adx_values[-1] / 100.0) # Normalize ADX strength to 0-1 range
            ))

        # Check for Sell Signal
        # Short MA crosses below Long MA AND ADX is above threshold
        elif (short_ma[-2] >= long_ma[-2] and short_ma[-1] < long_ma[-1] and adx_values[-1] > ADX_THRESHOLD):
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamps[-1],
                price=close_prices[-1],
                rule_id=RULE_ID,
                confidence=min(1.0, adx_values[-1] / 100.0) # Normalize ADX strength to 0-1 range
            ))

    return signals