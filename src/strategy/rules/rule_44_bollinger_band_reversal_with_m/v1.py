from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

# --- Parameters ---
BB_period = 20
BB_std_dev = 2
MFI_period = 14
MFI_oversold = 20
MFI_overbought = 80
Vol_SMA_period = 20
Vol_multiplier = 1.5
Fast_EMA_period = 10
Slow_EMA_period = 30

# Minimum data required for all indicators to have at least two valid points
# for current and previous checks (e.g., MFI_value[0] and MFI_value[1]).
# The slowest indicator is Slow_EMA (30 periods). For MFI, we need MFI_period + 1
# candles to get both current and previous MFI values.
# Max(Slow_EMA_period, MFI_period + 1) = Max(30, 14 + 1) = 30.
MIN_CANDLES_REQUIRED = max(Slow_EMA_period, MFI_period + 1)


# --- Helper Functions (Numpy-based) ---

def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average using numpy.
    Returns an array of length `len(data) - period + 1`, aligned to the right.
    """
    if len(data) < period:
        return np.array([])
    weights = np.ones(period) / period
    return np.convolve(data, weights, mode='valid')

def _stddev(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Rolling Standard Deviation using numpy.
    Returns an array of length `len(data) - period + 1`, aligned to the right.
    """
    if len(data) < period:
        return np.array([])
    std_devs = [np.std(data[i - period + 1 : i + 1]) for i in range(period - 1, len(data))]
    return np.array(std_devs)

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Exponential Moving Average using numpy.
    Returns an array of length `len(data) - period + 1`, aligned to the right.
    Initializes the first EMA value with the SMA of the first 'period' values.
    """
    if len(data) < period:
        return np.array([])
    alpha = 2 / (period + 1)
    ema_values = np.zeros_like(data, dtype=float)
    
    # Initialize the first EMA value with SMA of the first 'period' values
    ema_values[period - 1] = np.mean(data[:period])

    # Calculate subsequent EMA values
    for i in range(period, len(data)):
        ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
    
    return ema_values[period-1:] # Return valid EMA values from the first full period onwards

def _mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Calculates Money Flow Index (MFI) using numpy.
    Returns an array of length `len(data) - period + 1`, aligned to the right.
    """
    if len(high) < period:
        return np.array([])

    tp = (high + low + close) / 3
    money_flow = tp * volume

    pmf = np.zeros_like(money_flow)
    nmf = np.zeros_like(money_flow)

    # Calculate positive and negative money flow based on TP change
    for i in range(1, len(tp)):
        if tp[i] > tp[i-1]:
            pmf[i] = money_flow[i]
        elif tp[i] < tp[i-1]:
            nmf[i] = money_flow[i]
    
    # MFI values will be calculated for `len(close) - period + 1` data points.
    mfi_values = np.zeros(len(close) - period + 1, dtype=float)

    # Calculate rolling sums for PMF and NMF
    for i in range(len(close) - period + 1):
        window_start_idx = i
        window_end_idx = i + period - 1
        
        pos_flow_sum = np.sum(pmf[window_start_idx : window_end_idx + 1])
        neg_flow_sum = np.sum(nmf[window_start_idx : window_end_idx + 1])
        
        if neg_flow_sum == 0:
            # If no negative money flow, MFI is 100 (if positive flow exists) or 50 (if no flow at all)
            mfi_values[i] = 100.0 if pos_flow_sum > 0 else 50.0
        else:
            mfr = pos_flow_sum / neg_flow_sum
            mfi_values[i] = 100 - (100 / (1 + mfr))
            
    return mfi_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm

        if len(candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract data for numpy arrays
        close_prices = np.array([c.close for c in candles])
        high_prices = np.array([c.high for c in candles])
        low_prices = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        timestamps = [c.hour for c in candles]

        # --- Calculate Indicators ---
        # All helper functions return arrays aligned to the right (latest data point).
        # The last element of the returned array corresponds to the last candle in `candles`.

        bb_ma = _sma(close_prices, BB_period)
        bb_std = _stddev(close_prices, BB_period)
        mfi_vals = _mfi(high_prices, low_prices, close_prices, volumes, MFI_period)
        vol_sma = _sma(volumes, Vol_SMA_period)
        fast_ema = _ema(close_prices, Fast_EMA_period)
        slow_ema = _ema(close_prices, Slow_EMA_period)

        # Ensure all indicator calculations were successful and have at least 2 points for checks
        # (e.g., MFI turning upward/downward requires current and previous values).
        # MIN_CANDLES_REQUIRED should generally prevent empty arrays, but this is a safeguard.
        if any(len(arr) < 2 for arr in [bb_ma, bb_std, mfi_vals, vol_sma, fast_ema, slow_ema]):
            # The only exception is slow_ema, if len(candles) == Slow_EMA_period, it will have length 1.
            # But mfi_vals needs length >= 2.
            # Let's check for length 1 for slow_ema, and length 2 for mfi_vals.
            if len(slow_ema) < 1 or len(mfi_vals) < 2:
                continue
            # For other indicators, as their periods are smaller than Slow_EMA_period,
            # they will have at least 1 value (if len(candles) >= their period)
            # and usually more, so length 2 check is generally good.
            # For `bb_ma`, `bb_std`, `vol_sma`, `fast_ema`, if `len(candles) == 30`,
            # their lengths will be `30-20+1=11`, `30-20+1=11`, `30-20+1=11`, `30-10+1=21` respectively, all >= 2.
            # So the only specific checks needed are `len(slow_ema) < 1` and `len(mfi_vals) < 2`.
            # The condition `any(len(arr) < 2 for arr in [bb_ma, bb_std, mfi_vals, vol_sma, fast_ema, slow_ema])`
            # will correctly catch if `len(slow_ema)` is 0 or 1.
            # If `len(slow_ema)` is 1, it will be caught by this `if` statement.
            # This needs to be refined. `slow_ema_current` is `slow_ema[-1]`. If `len(slow_ema)` is 1, this works.
            # `mfi_value_previous` requires `len(mfi_vals) >= 2`.
            # So the minimum check is `len(slow_ema) < 1` OR `len(mfi_vals) < 2`.
            if len(slow_ema) < 1 or len(mfi_vals) < 2:
                continue


        # Extract latest values (corresponding to `candles[-1]`)
        current_close = close_prices[-1]
        current_volume = volumes[-1]
        current_timestamp = timestamps[-1]

        # Bollinger Bands
        upper_band = bb_ma[-1] + (bb_std[-1] * BB_std_dev)
        lower_band = bb_ma[-1] - (bb_std[-1] * BB_std_dev)

        # MFI (current and previous values for trend detection)
        mfi_value_current = mfi_vals[-1]
        mfi_value_previous = mfi_vals[-2]

        # Volume
        vol_sma_current = vol_sma[-1]
        is_high_volume = current_volume > (Vol_multiplier * vol_sma_current)

        # EMAs
        fast_ema_current = fast_ema[-1]
        slow_ema_current = slow_ema[-1]

        # --- Buy Signal Conditions ---
        if (current_close < lower_band and
            mfi_value_current < MFI_oversold and
            mfi_value_current > mfi_value_previous and  # MFI turning upward
            is_high_volume and
            fast_ema_current > slow_ema_current):  # Uptrend alignment
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id="8a653526-45c9-4f4d-8d4d-c58810043a05"
            ))

        # --- Sell Signal Conditions ---
        elif (current_close > upper_band and
              mfi_value_current > MFI_overbought and
              mfi_value_current < mfi_value_previous and  # MFI turning downward
              is_high_volume and
              fast_ema_current < slow_ema_current):  # Downtrend alignment
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id="8a653526-45c9-4f4d-8d4d-c58810043a05"
            ))

    return signals