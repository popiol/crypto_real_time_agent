from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Parameters
BB_PERIOD = 20
BB_STD_DEV = 2
MFI_PERIOD = 14
MFI_OVERSOLD_THRESHOLD = 20
MFI_OVERBOUGHT_THRESHOLD = 80
VOLUME_SMA_PERIOD = 20
BBW_EXPANSION_THRESHOLD = 0.005 # Percentage change for BBW expansion

# Minimum number of candles required for calculations
# We need `period` candles to calculate an indicator for the last candle in a window.
# To get *current* and *previous* indicator values, we need `period + 1` candles.
# We take the maximum of all periods and add 1.
MIN_CANDLES_REQUIRED = max(BB_PERIOD, MFI_PERIOD, VOLUME_SMA_PERIOD) + 1

def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < period:
        return np.array([])
    return np.convolve(data, np.ones(period), 'valid') / period

def _std_dev(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Standard Deviation for a rolling window."""
    if len(data) < period:
        return np.array([])
    result = np.zeros(len(data) - period + 1)
    for i in range(len(result)):
        result[i] = np.std(data[i : i + period])
    return result

def _mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Calculates Money Flow Index (MFI)."""
    if len(high) < period:
        return np.array([])

    typical_price = (high + low + close) / 3
    
    # Calculate Raw Money Flow
    money_flow = typical_price * volume

    # Calculate Positive and Negative Money Flow
    # Initialize arrays with zeros, the first element (index 0) will remain 0
    # as there's no previous typical price to compare with.
    positive_money_flow = np.zeros_like(typical_price)
    negative_money_flow = np.zeros_like(typical_price)

    # Compare typical price with previous day's typical price
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_money_flow[i] = money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_money_flow[i] = money_flow[i]
            
    # Calculate MFI over the specified period using rolling sums
    mfi_values = np.zeros(len(typical_price) - period + 1)
    for i in range(len(mfi_values)):
        period_positive_mf_sum = np.sum(positive_money_flow[i : i + period])
        period_negative_mf_sum = np.sum(negative_money_flow[i : i + period])

        if period_negative_mf_sum == 0:
            mfi_values[i] = 100.0 # If no negative money flow, MFI is 100 (strong buying pressure)
        else:
            money_ratio = period_positive_mf_sum / period_negative_mf_sum
            mfi_values[i] = 100 - (100 / (1 + money_ratio))
            
    return mfi_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    rule_id = "2ae372db-ed2d-4d51-b210-fe7a419a0bf9"

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract relevant data as numpy arrays for efficient calculation
        # Assuming warm_candles are sorted chronologically
        closes = np.array([c.close for c in warm_candles])
        highs = np.array([c.high for c in warm_candles])
        lows = np.array([c.low for c in warm_candles])
        volumes = np.array([c.volume for c in warm_candles])
        timestamps = np.array([c.hour for c in warm_candles])

        # --- Calculate Bollinger Bands ---
        # SMA and STD_DEV for BB_PERIOD. These arrays are aligned to the end of their windows.
        bb_sma_arr = _sma(closes, BB_PERIOD)
        bb_std_dev_arr = _std_dev(closes, BB_PERIOD)

        if len(bb_sma_arr) < 2: # Need at least 2 points for current and previous BBW
            continue
        
        upper_band_arr = bb_sma_arr + (bb_std_dev_arr * BB_STD_DEV)
        lower_band_arr = bb_sma_arr - (bb_std_dev_arr * BB_STD_DEV)

        # --- Calculate MFI ---
        mfi_arr = _mfi(highs, lows, closes, volumes, MFI_PERIOD)
        if len(mfi_arr) < 2: # Need at least 2 points for current and previous MFI
            continue

        # --- Calculate Volume SMA ---
        volume_sma_arr = _sma(volumes, VOLUME_SMA_PERIOD)
        if len(volume_sma_arr) < 1: # Need at least 1 point for current volume SMA
            continue

        # --- Calculate Bollinger Band Width (BBW) ---
        # BBW is (UPPER - LOWER) / SMA. The SMA here is the BB_PERIOD SMA.
        bbw_arr = (upper_band_arr - lower_band_arr) / bb_sma_arr
        # bbw_arr length is same as bb_sma_arr, already checked for < 2

        # --- Aligning and extracting current and previous indicator values ---
        # The indicator arrays have different lengths based on their periods.
        # We need to find the common valid indices for the *last* and *second to last* candles.
        # All indicator arrays' `[-1]` element corresponds to the window ending at `closes[-1]`.
        # All indicator arrays' `[-2]` element corresponds to the window ending at `closes[-2]`.
        # This alignment is crucial and assumed by the `_sma`, `_std_dev`, `_mfi` functions.

        # Current candle's data (last available full set of indicators)
        current_close = closes[-1]
        current_volume = volumes[-1]
        current_timestamp = timestamps[-1]

        # Ensure all required indicator arrays have at least 2 elements for current and previous
        if len(bb_sma_arr) < 2 or len(mfi_arr) < 2 or len(volume_sma_arr) < 1:
            continue
        
        upper_band_current = upper_band_arr[-1]
        lower_band_current = lower_band_arr[-1]
        mfi_current = mfi_arr[-1]
        volume_sma_current = volume_sma_arr[-1]
        bbw_current = bbw_arr[-1]

        # Previous values (second to last available full set of indicators)
        mfi_previous = mfi_arr[-2]
        bbw_previous = bbw_arr[-2]
        
        # --- Buy Signal Logic ---
        buy_condition_1 = current_close < lower_band_current
        buy_condition_2 = mfi_current < MFI_OVERSOLD_THRESHOLD
        buy_condition_3 = mfi_current > mfi_previous # MFI turning upwards
        buy_condition_4 = current_volume > volume_sma_current
        buy_condition_5 = bbw_current > bbw_previous * (1 + BBW_EXPANSION_THRESHOLD)

        if buy_condition_1 and buy_condition_2 and buy_condition_3 and buy_condition_4 and buy_condition_5:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id=rule_id
            ))

        # --- Sell Signal Logic ---
        sell_condition_1 = current_close > upper_band_current
        sell_condition_2 = mfi_current > MFI_OVERBOUGHT_THRESHOLD
        sell_condition_3 = mfi_current < mfi_previous # MFI turning downwards
        sell_condition_4 = current_volume > volume_sma_current
        sell_condition_5 = bbw_current > bbw_previous * (1 + BBW_EXPANSION_THRESHOLD)

        if sell_condition_1 and sell_condition_2 and sell_condition_3 and sell_condition_4 and sell_condition_5:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id=rule_id
            ))

    return signals