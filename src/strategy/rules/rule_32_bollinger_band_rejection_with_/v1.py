from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Constants for the rule ---
BB_WINDOW = 20
BB_NUM_STD_DEV = 2.0
MFI_WINDOW = 14
VOLUME_SMA_WINDOW = 20
MFI_OVERSOLD = 30
MFI_OVERBOUGHT = 70
VOLUME_MULTIPLIER = 1.5
WICK_PERCENTAGE = 0.3

# Minimum number of candles required to calculate all indicators
MIN_CANDLES = max(BB_WINDOW, MFI_WINDOW, VOLUME_SMA_WINDOW)

# --- Helper Functions ---

def _calculate_sma(data: np.ndarray, window: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < window:
        return np.full(len(data), np.nan)
    
    # Using np.convolve for efficiency
    weights = np.ones(window) / window
    sma = np.convolve(data, weights, mode='valid')
    
    # Pad the beginning with NaNs to match original data length
    return np.concatenate((np.full(window - 1, np.nan), sma))

def _calculate_bollinger_bands(close: np.ndarray, window: int, num_std_dev: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates Bollinger Bands (SMA, Upper, Lower)."""
    if len(close) < window:
        return np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan)

    sma = _calculate_sma(close, window)
    
    # Calculate rolling standard deviation
    std_dev = np.full(len(close), np.nan)
    for i in range(window - 1, len(close)):
        std_dev[i] = np.std(close[i - window + 1 : i + 1])

    upper_band = sma + std_dev * num_std_dev
    lower_band = sma - std_dev * num_std_dev
    return sma, upper_band, lower_band

def _calculate_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    """Calculates Money Flow Index (MFI)."""
    if len(high) < window:
        return np.full(len(high), np.nan)

    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_money_flow = np.zeros_like(money_flow)
    negative_money_flow = np.zeros_like(money_flow)

    # Calculate positive and negative money flow
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_money_flow[i] = money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_money_flow[i] = money_flow[i]

    mfi = np.full(len(high), np.nan)
    
    # Calculate MFI over the window
    for i in range(window - 1, len(high)):
        pmf_sum = np.sum(positive_money_flow[i - window + 1 : i + 1])
        nmf_sum = np.sum(negative_money_flow[i - window + 1 : i + 1])

        if nmf_sum == 0:
            # If there's no negative money flow, MFI is typically 100
            mfi[i] = 100.0
        else:
            money_flow_ratio = pmf_sum / nmf_sum
            mfi[i] = 100 - (100 / (1 + money_flow_ratio))
            
    return mfi

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    rule_id = "982c752a-e97d-4dd7-972b-5601042399b6"

    for pair, pair_data in data.items():
        candles = pair_data.warm

        if len(candles) < MIN_CANDLES:
            continue

        # Convert list of WarmCandle objects to numpy arrays for efficient calculations
        timestamps = np.array([c.hour for c in candles])
        open_prices = np.array([c.open_price for c in candles])
        high_prices = np.array([c.high for c in candles])
        low_prices = np.array([c.low for c in candles])
        close_prices = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Calculate indicators
        _, bb_upper, bb_lower = _calculate_bollinger_bands(close_prices, BB_WINDOW, BB_NUM_STD_DEV)
        mfi = _calculate_mfi(high_prices, low_prices, close_prices, volumes, MFI_WINDOW)
        volume_sma = _calculate_sma(volumes, VOLUME_SMA_WINDOW)

        # Iterate through candles starting from the point where all indicators have valid values
        # The pseudocode iterates from 1, but we need to ensure all indicators have valid values.
        # The `_calculate_` functions already pad with NaN, so we can iterate from the first non-NaN index.
        start_idx = MIN_CANDLES - 1 # Adjust for 0-based indexing

        for i in range(start_idx, len(candles)):
            current_candle = candles[i]
            
            current_open = open_prices[i]
            current_high = high_prices[i]
            current_low = low_prices[i]
            current_close = close_prices[i]
            current_volume = volumes[i]

            # Ensure all indicators have valid values for the current index
            if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
                np.isnan(mfi[i]) or np.isnan(volume_sma[i])):
                continue

            # Calculate candle body and wick lengths
            candle_range = current_high - current_low
            
            # Handle cases where candle_range might be zero to avoid division by zero
            if candle_range == 0:
                continue

            upper_wick = current_high - max(current_open, current_close)
            lower_wick = min(current_open, current_close) - current_low

            # Buy Signal Conditions
            # Price touched/breached lower band but closed above it
            if current_low <= bb_lower[i] and current_close > bb_lower[i]:
                # MFI is oversold
                if mfi[i] < MFI_OVERSOLD:
                    # High volume
                    if current_volume > VOLUME_MULTIPLIER * volume_sma[i]:
                        # Significant lower wick
                        if lower_wick > WICK_PERCENTAGE * candle_range:
                            signals.append(BuySignal(
                                pair=pair,
                                timestamp=current_candle.hour,
                                price=current_candle.close,
                                rule_id=rule_id,
                                confidence=1.0 
                            ))

            # Sell Signal Conditions
            # Price touched/breached upper band but closed below it
            if current_high >= bb_upper[i] and current_close < bb_upper[i]:
                # MFI is overbought
                if mfi[i] > MFI_OVERBOUGHT:
                    # High volume
                    if current_volume > VOLUME_MULTIPLIER * volume_sma[i]:
                        # Significant upper wick
                        if upper_wick > WICK_PERCENTAGE * candle_range:
                            signals.append(SellSignal(
                                pair=pair,
                                timestamp=current_candle.hour,
                                price=current_candle.close,
                                rule_id=rule_id,
                                confidence=1.0 
                            ))
    return signals