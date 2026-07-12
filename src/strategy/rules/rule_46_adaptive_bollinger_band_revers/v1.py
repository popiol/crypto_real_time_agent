from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal

# Rule ID from the idea_id
RULE_ID = "14f99bae-60f1-4aec-9102-a3f4f7516458"

# Parameters
BB_PERIOD = 20
BB_STD_DEV = 2
MFI_PERIOD = 14
RSI_PERIOD = 14
MFI_OVERSOLD_THRESHOLD = 20
MFI_OVERBOUGHT_THRESHOLD = 80
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70
VOLUME_SMA_PERIOD = 20
VOLATILITY_FACTOR = 1.5

# Minimum candles required for all indicators to have at least two valid values (current and previous)
# The longest period is 20. We need 20 candles to calculate the first valid indicator value,
# and one more candle to have a "current" and "previous" value for comparison.
MIN_CANDLES = max(BB_PERIOD, MFI_PERIOD, RSI_PERIOD, VOLUME_SMA_PERIOD) + 1

# --- Helper Functions for Indicators ---

def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average, returning an array of the same length as data,
    padded with NaNs at the beginning if data is insufficient for a full period.
    """
    if len(data) < period:
        return np.full(len(data), np.nan)
    
    weights = np.ones(period) / period
    sma_values = np.convolve(data, weights, mode='valid')
    
    nan_padding = np.full(period - 1, np.nan)
    return np.concatenate((nan_padding, sma_values))

def _std_dev(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Standard Deviation, returning an array of the same length as data,
    padded with NaNs at the beginning.
    """
    if len(data) < period:
        return np.full(len(data), np.nan)
    
    std_values = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        std_values[i] = np.std(data[i - period + 1 : i + 1])
    return std_values

def _atr(high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates Average True Range, returning an array of the same length as prices,
    padded with NaNs at the beginning.
    """
    if len(high_prices) < 2: # Need at least two candles for the first True Range calculation
        return np.full(len(high_prices), np.nan)

    tr = np.full(len(high_prices), np.nan)
    for i in range(1, len(high_prices)):
        high_i = high_prices[i]
        low_i = low_prices[i]
        close_prev = close_prices[i-1]
        tr[i] = max(high_i - low_i, abs(high_i - close_prev), abs(low_i - close_prev))
    
    # ATR is the SMA of True Range. The _sma function handles NaN padding.
    return _sma(tr, period)

def _mfi(high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray, volumes: np.ndarray, period: int) -> np.ndarray:
    """Calculates Money Flow Index, returning an array of the same length as prices,
    padded with NaNs at the beginning.
    """
    if len(close_prices) < period + 1:
        return np.full(len(close_prices), np.nan)

    typical_prices = (high_prices + low_prices + close_prices) / 3
    raw_money_flow = typical_prices * volumes

    pos_money_flow = np.zeros(len(close_prices))
    neg_money_flow = np.zeros(len(close_prices))

    for i in range(1, len(close_prices)):
        if typical_prices[i] > typical_prices[i-1]:
            pos_money_flow[i] = raw_money_flow[i]
        elif typical_prices[i] < typical_prices[i-1]:
            neg_money_flow[i] = raw_money_flow[i]

    mfi_values = np.full(len(close_prices), np.nan)
    for i in range(period, len(close_prices)):
        period_pos_mf = np.sum(pos_money_flow[i - period + 1 : i + 1])
        period_neg_mf = np.sum(neg_money_flow[i - period + 1 : i + 1])

        if period_neg_mf == 0:
            mfi_values[i] = 100.0 # Extremely bullish if no negative money flow
        elif period_pos_mf == 0:
            mfi_values[i] = 0.0 # Extremely bearish if no positive money flow
        else:
            money_flow_ratio = period_pos_mf / period_neg_mf
            mfi_values[i] = 100 - (100 / (1 + money_flow_ratio))
    return mfi_values

def _rsi(close_prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates Relative Strength Index, returning an array of the same length as prices,
    padded with NaNs at the beginning.
    """
    if len(close_prices) < period + 1:
        return np.full(len(close_prices), np.nan)

    price_changes = np.diff(close_prices) # Array of length N-1
    gains = np.maximum(0, price_changes)
    losses = np.abs(np.minimum(0, price_changes))

    rsi_values = np.full(len(close_prices), np.nan)
    
    if len(gains) < period: # Not enough price changes for the initial period calculation
        return np.full(len(close_prices), np.nan)

    # Calculate initial average gain/loss using simple average for the first 'period' elements
    # These correspond to price changes for candles from index 1 to 'period' (0-indexed gains/losses)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # The first valid RSI value corresponds to the candle at index 'period' in original close_prices
    if avg_loss == 0:
        rsi_values[period] = 100.0 if avg_gain > 0 else 50.0 # Handle division by zero
    else:
        rs = avg_gain / avg_loss
        rsi_values[period] = 100 - (100 / (1 + rs))

    # Calculate subsequent RSI values using the smoothing formula (Wilder's smoothing)
    for i in range(period, len(gains)): # Iterate over price_changes starting from index 'period'
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        # The RSI value corresponds to the candle at index (i+1) in original close_prices
        if avg_loss == 0:
            rsi_values[i+1] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i+1] = 100 - (100 / (1 + rs))
    
    return rsi_values

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm
        
        if len(candles) < MIN_CANDLES:
            continue

        # Extract required candle data into numpy arrays
        close_prices = np.array([c.close for c in candles])
        high_prices = np.array([c.high for c in candles])
        low_prices = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        timestamps = [c.hour for c in candles]

        # Calculate Indicators
        bb_middle = _sma(close_prices, BB_PERIOD)
        bb_std = _std_dev(close_prices, BB_PERIOD)
        bb_upper = bb_middle + BB_STD_DEV * bb_std
        bb_lower = bb_middle - BB_STD_DEV * bb_std
        
        mfi = _mfi(high_prices, low_prices, close_prices, volumes, MFI_PERIOD)
        rsi = _rsi(close_prices, RSI_PERIOD)
        # ATR period is not specified in pseudocode, using BB_PERIOD (20) as a reasonable default
        atr = _atr(high_prices, low_prices, close_prices, BB_PERIOD) 

        volume_sma = _sma(volumes, VOLUME_SMA_PERIOD)

        # Check if the latest two indicator values are valid (not NaN) for current and previous comparisons
        indicator_values = [
            bb_lower[-1], bb_lower[-2], bb_upper[-1], bb_upper[-2],
            mfi[-1], mfi[-2], rsi[-1], rsi[-2],
            atr[-1], volume_sma[-1]
        ]
        if any(np.isnan(val) for val in indicator_values):
            continue

        # Get current and previous values for conditions
        current_close = close_prices[-1]
        previous_close = close_prices[-2]
        current_volume = volumes[-1]
        current_timestamp = timestamps[-1]

        bb_lower_current = bb_lower[-1]
        bb_lower_previous = bb_lower[-2]
        bb_upper_current = bb_upper[-1]
        bb_upper_previous = bb_upper[-2]
        
        mfi_current = mfi[-1]
        mfi_previous = mfi[-2]
        rsi_current = rsi[-1]
        rsi_previous = rsi[-2]
        
        atr_current = atr[-1]
        volume_sma_current = volume_sma[-1]

        # Dynamic Volume Threshold calculation
        # Add a small epsilon to avoid division by zero if current_close is exactly 0
        denominator = current_close if current_close != 0 else 0.0000001 
        dynamic_volume_threshold = volume_sma_current * (1 + (atr_current / denominator) * VOLATILITY_FACTOR)

        # Buy Signal Conditions
        # Price breaches lower BB, then closes back inside
        buy_condition_bb = (previous_close < bb_lower_previous and current_close > bb_lower_current)
        # MFI oversold and turning up
        buy_condition_mfi = (mfi_current < MFI_OVERSOLD_THRESHOLD and mfi_current > mfi_previous)
        # RSI oversold and turning up
        buy_condition_rsi = (rsi_current < RSI_OVERSOLD_THRESHOLD and rsi_current > rsi_previous)
        # Volume confirmation
        buy_condition_volume = (current_volume > dynamic_volume_threshold)

        if (buy_condition_bb and 
            buy_condition_mfi and 
            buy_condition_rsi and 
            buy_condition_volume):
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id=RULE_ID
            ))

        # Sell Signal Conditions (symmetrical to Buy)
        # Price breaches upper BB, then closes back inside
        sell_condition_bb = (previous_close > bb_upper_previous and current_close < bb_upper_current)
        # MFI overbought and turning down
        sell_condition_mfi = (mfi_current > MFI_OVERBOUGHT_THRESHOLD and mfi_current < mfi_previous)
        # RSI overbought and turning down
        sell_condition_rsi = (rsi_current > RSI_OVERBOUGHT_THRESHOLD and rsi_current < rsi_previous)
        # Volume confirmation
        sell_condition_volume = (current_volume > dynamic_volume_threshold)

        if (sell_condition_bb and 
            sell_condition_mfi and 
            sell_condition_rsi and 
            sell_condition_volume):
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id=RULE_ID
            ))

    return signals