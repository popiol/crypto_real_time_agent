from __future__ import annotations
import numpy as np
from datetime import datetime
from typing import List, Union

from src.agent.models import BuySignal, SellSignal, MarketData, WarmCandle

# Parameters
BB_PERIOD = 20
BB_STD_DEV = 2
MFI_PERIOD = 14
MFI_OVERSOLD = 20
MFI_OVERBOUGHT = 80
VOLUME_MA_PERIOD = 20
# Adjusted EMA periods due to WarmCandle data constraint (max 24 entries).
# Original pseudocode had SHORT_EMA_PERIOD = 20, LONG_EMA_PERIOD = 100.
# With max 24 candles, a period of 100 is impossible.
# These adjustments make the rule executable with available data,
# but might reduce the "multi-timeframe" distinction.
SHORT_EMA_PERIOD = 10
LONG_EMA_PERIOD = 20

# Minimum number of candles required for calculations.
# This accounts for the largest period (LONG_EMA_PERIOD or BB_PERIOD)
# plus additional candles needed for previous values (e.g., MFI_PREV, LONG_EMA[2]).
# Max_period + 2 ensures we have enough data for `long_ema_arr[-3]` and `mfi_arr[-2]`.
MIN_CANDLES_REQUIRED = max(BB_PERIOD, MFI_PERIOD, VOLUME_MA_PERIOD, SHORT_EMA_PERIOD, LONG_EMA_PERIOD) + 2

def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < period:
        return np.array([])
    return np.convolve(data, np.ones(period) / period, mode='valid')

def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Exponential Moving Average."""
    if len(data) < period:
        return np.array([])
    
    ema = np.zeros_like(data, dtype=float)
    
    # Initialize the first EMA value with an SMA of the first `period` values
    initial_sma = _calculate_sma(data[:period], period)
    if len(initial_sma) == 0:
        return np.array([]) # Not enough data even for initial SMA
    
    ema[period - 1] = initial_sma[-1]

    multiplier = 2 / (period + 1)
    for i in range(period, len(data)):
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
    
    # Return only the valid EMA values (from index period-1 onwards)
    return ema[period - 1:]

def _calculate_bollinger_bands(close_prices: np.ndarray, period: int, std_dev_multiplier: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates Bollinger Bands (Lower, Middle, Upper)."""
    if len(close_prices) < period:
        return np.array([]), np.array([]), np.array([])
    
    middle_band = _calculate_sma(close_prices, period)
    
    # Calculate rolling standard deviation
    std_devs = np.array([np.std(close_prices[i : i + period]) for i in range(len(close_prices) - period + 1)])
    
    upper_band = middle_band + (std_devs * std_dev_multiplier)
    lower_band = middle_band - (std_devs * std_dev_multiplier)
    
    return lower_band, middle_band, upper_band

def _calculate_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Calculates Money Flow Index."""
    if len(high) < period:
        return np.array([])

    tp = (high + low + close) / 3  # Typical Price
    raw_mf = tp * volume           # Raw Money Flow

    mfi_values = []
    # Iterate from the first point where MFI can be calculated (needs `period` previous bars)
    for i in range(period, len(close)):
        pos_mf_sum = 0.0
        neg_mf_sum = 0.0
        
        # Calculate sums of positive and negative money flow over the `period`
        # The range is `i-period+1` to `i`, inclusive.
        # This requires `tp[k]` and `tp[k-1]`, so `k` starts from `i-period+1`.
        for k in range(i - period + 1, i + 1):
            if k == 0: # Cannot compare tp[0] with tp[-1]
                continue 
            
            if tp[k] > tp[k-1]:
                pos_mf_sum += raw_mf[k]
            elif tp[k] < tp[k-1]:
                neg_mf_sum += raw_mf[k]
        
        # Handle division by zero for Money Flow Ratio
        if neg_mf_sum == 0:
            mfr = 200.0 if pos_mf_sum > 0 else 0.0 # Assign a large value if only positive flow, 0 if no flow
        else:
            mfr = pos_mf_sum / neg_mf_sum
        
        mfi = 100 - (100 / (1 + mfr))
        mfi_values.append(mfi)
        
    return np.array(mfi_values)

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm
        
        if len(candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract relevant data for calculations as numpy arrays
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        timestamps = [c.hour for c in candles]

        # Calculate all required indicators
        lower_bb_arr, _, upper_bb_arr = _calculate_bollinger_bands(closes, BB_PERIOD, BB_STD_DEV)
        mfi_arr = _calculate_mfi(highs, lows, closes, volumes, MFI_PERIOD)
        volume_ma_arr = _calculate_sma(volumes, VOLUME_MA_PERIOD)
        short_ema_arr = _calculate_ema(closes, SHORT_EMA_PERIOD)
        long_ema_arr = _calculate_ema(closes, LONG_EMA_PERIOD)

        # Ensure all indicator arrays have enough data points for the latest candle
        # and for previous values (MFI_PREV, LONG_EMA[2])
        if (len(lower_bb_arr) == 0 or len(upper_bb_arr) == 0 or
            len(mfi_arr) < 2 or # Need current and previous MFI
            len(volume_ma_arr) == 0 or
            len(short_ema_arr) == 0 or
            len(long_ema_arr) < 3): # Need current and two previous EMAs for trend
            continue

        # Get the latest values for the current candle
        current_close = closes[-1]
        current_volume = volumes[-1]
        current_timestamp = timestamps[-1]

        current_lower_bb = lower_bb_arr[-1]
        current_upper_bb = upper_bb_arr[-1]
        current_mfi = mfi_arr[-1]
        prev_mfi = mfi_arr[-2]
        current_volume_ma = volume_ma_arr[-1]
        current_short_ema = short_ema_arr[-1]
        
        # Long EMA values for trend determination (current, prev1, prev2)
        long_ema_0 = long_ema_arr[-1]
        long_ema_1 = long_ema_arr[-2]
        long_ema_2 = long_ema_arr[-3]

        # Determine Long-Term Trend based on LONG_EMA slope
        is_long_term_uptrend = (long_ema_0 > long_ema_1) and (long_ema_1 > long_ema_2)
        is_long_term_downtrend = (long_ema_0 < long_ema_1) and (long_ema_1 < long_ema_2)
        is_long_term_neutral = not is_long_term_uptrend and not is_long_term_downtrend
        
        # Buy Signal Logic
        if (
            current_close < current_lower_bb and             # Price closes below lower BB (mean reversion trigger)
            current_mfi < MFI_OVERSOLD and                   # MFI is oversold
            current_mfi > prev_mfi and                       # MFI is turning upward (reversal confirmation)
            current_volume > current_volume_ma and           # High volume (conviction)
            current_close > current_short_ema and            # Current price is above short-term EMA (local strength)
            (is_long_term_uptrend or is_long_term_neutral)   # Longer-term trend is upward or neutral
        ):
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id="2a5a0d96-18e3-4288-bbfa-acd58fcb00f7"
            ))

        # Sell Signal Logic
        if (
            current_close > current_upper_bb and             # Price closes above upper BB (mean reversion trigger)
            current_mfi > MFI_OVERBOUGHT and                 # MFI is overbought
            current_mfi < prev_mfi and                       # MFI is turning downward (reversal confirmation)
            current_volume > current_volume_ma and           # High volume (conviction)
            current_close < current_short_ema and            # Current price is below short-term EMA (local weakness)
            (is_long_term_downtrend or is_long_term_neutral) # Longer-term trend is downward or neutral
        ):
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id="2a5a0d96-18e3-4288-bbfa-acd58fcb00f7"
            ))
            
    return signals