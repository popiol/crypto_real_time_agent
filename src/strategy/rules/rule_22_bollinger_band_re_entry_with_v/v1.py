from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Constants ---
PERIOD_BB = 20
STD_DEV_BB = 2.0
PERIOD_MFI = 14
MFI_OVERSOLD = 20
MFI_OVERBOUGHT = 80
VOLUME_SPIKE_MULTIPLIER = 1.5

# Minimum number of candles required for calculations.
# To obtain a valid 'current' and 'previous' value for an indicator
# with a window 'W', we need at least 'W + 1' candles.
# For example, a 20-period SMA will have its first valid value at index 19 (0-indexed).
# To have a 'previous' valid value, we need index 18 to also be valid,
# which requires at least 20 + 1 = 21 candles in total.
MIN_CANDLES = max(PERIOD_BB, PERIOD_MFI) + 1

# --- Helper Functions for Indicators ---

def _sma(data: np.ndarray, window: int) -> np.ndarray:
    """Calculates the Simple Moving Average for a given data array."""
    if len(data) < window:
        return np.full(len(data), np.nan)
    # Use convolution for efficient SMA calculation over the full series
    weights = np.ones(window) / window
    sma_values = np.convolve(data, weights, mode='valid')
    # Pad with NaNs at the beginning to align the output length with the input data
    return np.concatenate((np.full(window - 1, np.nan), sma_values))

def _std_dev(data: np.ndarray, window: int) -> np.ndarray:
    """Calculates the rolling Standard Deviation for a given data array."""
    if len(data) < window:
        return np.full(len(data), np.nan)
    # Calculate rolling standard deviation using a list comprehension for clarity
    std_devs = np.array([np.std(data[i-window+1:i+1]) for i in range(window - 1, len(data))])
    # Pad with NaNs at the beginning
    return np.concatenate((np.full(window - 1, np.nan), std_devs))

def _mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    """Calculates the Money Flow Index (MFI)."""
    if len(high) < window:
        return np.full(len(high), np.nan)

    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_money_flow = np.zeros_like(money_flow)
    negative_money_flow = np.zeros_like(money_flow)

    # Calculate positive and negative money flow based on typical price changes
    # Start from the second candle as comparison to previous is needed
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_money_flow[i] = money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_money_flow[i] = money_flow[i]
        # If typical_price[i] == typical_price[i-1], both positive_money_flow[i] and negative_money_flow[i] remain 0

    mfi_values = np.full(len(high), np.nan)

    # Calculate MFI starting from the first index where a full window of data is available
    for i in range(window - 1, len(high)):
        pos_mf_sum = np.sum(positive_money_flow[i - window + 1 : i + 1])
        neg_mf_sum = np.sum(negative_money_flow[i - window + 1 : i + 1])

        if neg_mf_sum == 0:
            # If there's no negative money flow, MFI tends to 100 (extreme overbought)
            # unless there's also no positive money flow, in which case it's neutral (50).
            if pos_mf_sum == 0:
                mfi_values[i] = 50.0  # Neutral, no money flow in either direction
            else:
                mfi_values[i] = 100.0
        else:
            money_flow_ratio = pos_mf_sum / neg_mf_sum
            mfi_values[i] = 100 - (100 / (1 + money_flow_ratio))
    return mfi_values

# --- Main Signal Function ---
def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Detects Bollinger Band re-entry with volume and MFI confirmation for buy/sell signals.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        if len(warm_candles) < MIN_CANDLES:
            continue

        # Extract necessary data into numpy arrays for efficient calculation
        close_prices = np.array([c.close for c in warm_candles])
        high_prices = np.array([c.high for c in warm_candles])
        low_prices = np.array([c.low for c in warm_candles])
        volumes = np.array([c.volume for c in warm_candles])
        timestamps = [c.hour for c in warm_candles]

        # Calculate indicators for the entire available candle history
        sma_bb = _sma(close_prices, PERIOD_BB)
        std_dev_bb = _std_dev(close_prices, PERIOD_BB)
        upper_bb = sma_bb + (STD_DEV_BB * std_dev_bb)
        lower_bb = sma_bb - (STD_DEV_BB * std_dev_bb)
        mfi = _mfi(high_prices, low_prices, close_prices, volumes, PERIOD_MFI)
        avg_volume = _sma(volumes, PERIOD_BB)  # Using SMA for average volume

        # Get the index for the current and previous candle (last two entries in the series)
        current_idx = len(warm_candles) - 1
        prev_idx = len(warm_candles) - 2

        # Extract all necessary values for the current and previous candles
        current_close = close_prices[current_idx]
        current_volume = volumes[current_idx]
        current_mfi = mfi[current_idx]
        current_lower_bb = lower_bb[current_idx]
        current_upper_bb = upper_bb[current_idx]
        current_avg_volume = avg_volume[current_idx]

        prev_close = close_prices[prev_idx]
        prev_mfi = mfi[prev_idx]
        prev_lower_bb = lower_bb[prev_idx]
        prev_upper_bb = upper_bb[prev_idx]

        # Check for NaN values in any of the critical indicators for the last two candles.
        # This ensures all calculations were successful and we have valid data.
        if any(np.isnan([current_close, current_volume, current_mfi, current_lower_bb, current_upper_bb, current_avg_volume,
                         prev_close, prev_mfi, prev_lower_bb, prev_upper_bb])):
            continue  # Skip if any critical value is NaN

        # --- Buy Signal Logic ---
        # Condition 1: Price was below the lower Bollinger Band, then re-entered above it.
        # Condition 2: Current volume is significantly higher than average volume.
        # Condition 3: MFI shows an upward turn from oversold territory.
        if (prev_close < prev_lower_bb and  # Price was below lower BB
            current_close > current_lower_bb and  # Price re-entered above lower BB
            current_volume > (current_avg_volume * VOLUME_SPIKE_MULTIPLIER) and  # Volume spike
            current_mfi > prev_mfi and  # MFI shows upward turn
            prev_mfi < MFI_OVERSOLD):  # MFI was in oversold territory
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamps[current_idx],
                price=current_close,
                rule_id="f3fac943-e09c-415b-ab7f-fdb8f7b5620e",
            ))

        # --- Sell Signal Logic ---
        # Condition 1: Price was above the upper Bollinger Band, then re-entered below it.
        # Condition 2: Current volume is significantly higher than average volume.
        # Condition 3: MFI shows a downward turn from overbought territory.
        elif (prev_close > prev_upper_bb and  # Price was above upper BB
              current_close < current_upper_bb and  # Price re-entered below upper BB
              current_volume > (current_avg_volume * VOLUME_SPIKE_MULTIPLIER) and  # Volume spike
              current_mfi < prev_mfi and  # MFI shows downward turn
              prev_mfi > MFI_OVERBOUGHT):  # MFI was in overbought territory
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamps[current_idx],
                price=current_close,
                rule_id="f3fac943-e09c-415b-ab7f-fdb8f7b5620e",
            ))

    return signals