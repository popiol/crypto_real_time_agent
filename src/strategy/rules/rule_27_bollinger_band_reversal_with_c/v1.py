from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle
from datetime import datetime

# --- Constants ---
RULE_ID = "ffc2dea0-0016-4adf-abe3-f38ada26273d"

BB_PERIOD = 20
BB_NUM_STD_DEV = 2

MFI_PERIOD = 14
MFI_OVERSOLD_THRESHOLD = 20
MFI_OVERBOUGHT_THRESHOLD = 80

VOLUME_SMA_PERIOD = 20

# Minimum candles required for all calculations (BB, MFI, Volume SMA)
# MFI needs period + 1 for price comparison (to detect price direction change)
MIN_CANDLES_REQUIRED = max(BB_PERIOD, MFI_PERIOD + 1, VOLUME_SMA_PERIOD)

# Candlestick pattern constants
CANDLE_BODY_RATIO_MAX = 0.3  # Body size relative to total range (high - low)
CANDLE_SHADOW_RATIO_MIN = 2.0 # Main shadow relative to body size
CANDLE_OPPOSITE_SHADOW_RATIO_MAX = 0.1 # Opposite shadow relative to body size
MIN_BODY_SIZE = 1e-9 # Minimum body size to avoid division by zero in ratio checks, and filter dojis

# --- Helper Functions ---

def _calculate_sma(data: np.ndarray, period: int) -> float:
    """Calculates Simple Moving Average for the last 'period' elements."""
    if len(data) < period:
        return np.nan
    return np.mean(data[-period:])

def _calculate_bollinger_bands(closes: np.ndarray, period: int, num_std_dev: float) -> tuple[float, float, float]:
    """Calculates Bollinger Bands (SMA, UBB, LBB) for the last 'period' closes."""
    if len(closes) < period:
        return np.nan, np.nan, np.nan
    
    # Calculate SMA for the last 'period' closes
    sma = _calculate_sma(closes, period)
    
    # Calculate Standard Deviation for the last 'period' closes
    std_dev = np.std(closes[-period:])
    
    ubb = sma + (num_std_dev * std_dev)
    lbb = sma - (num_std_dev * std_dev)
    
    return sma, ubb, lbb

def _calculate_mfi(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray, period: int) -> float:
    """Calculates Money Flow Index (MFI) for the last 'period' candles."""
    # Need at least period + 1 data points to calculate one MFI value
    if len(highs) < period + 1:
        return np.nan

    # Slice data to the required length for MFI calculation
    # MFI uses 'period' values for summation, but needs one prior value for price change comparison
    highs_slice = highs[-(period + 1):]
    lows_slice = lows[-(period + 1):]
    closes_slice = closes[-(period + 1):]
    volumes_slice = volumes[-(period + 1):]

    typical_prices = (highs_slice + lows_slice + closes_slice) / 3
    money_flow = typical_prices * volumes_slice

    positive_money_flow_series = np.zeros(len(typical_prices))
    negative_money_flow_series = np.zeros(len(typical_prices))

    for i in range(1, len(typical_prices)):
        if typical_prices[i] > typical_prices[i-1]:
            positive_money_flow_series[i] = money_flow[i]
        elif typical_prices[i] < typical_prices[i-1]:
            negative_money_flow_series[i] = money_flow[i]

    # Sum over the last 'period' elements of the money flow series
    # These sums are for the period ending with the last candle
    sum_positive_mf = np.sum(positive_money_flow_series[-period:])
    sum_negative_mf = np.sum(negative_money_flow_series[-period:])

    if sum_negative_mf == 0 and sum_positive_mf == 0:
        return 50.0 # Neutral if no flow in either direction
    elif sum_negative_mf == 0:
        return 100.0 # No negative flow, all positive
    elif sum_positive_mf == 0:
        return 0.0 # No positive flow, all negative
    else:
        money_ratio = sum_positive_mf / sum_negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

def _is_hammer(open_price: float, high: float, low: float, close: float) -> bool:
    """Checks if the given candle is a Hammer pattern."""
    body = abs(close - open_price)
    candle_range = high - low
    
    # Invalid candle if range is zero or body is too small to calculate ratios
    if candle_range == 0 or body < MIN_BODY_SIZE:
        return False

    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low

    # Conditions for a Hammer:
    # 1. Small body relative to total range
    # 2. Long lower shadow (at least 2x body)
    # 3. Very small or no upper shadow (less than 10% of body relative to body size)
    return (body < CANDLE_BODY_RATIO_MAX * candle_range and
            lower_shadow >= CANDLE_SHADOW_RATIO_MIN * body and
            upper_shadow < CANDLE_OPPOSITE_SHADOW_RATIO_MAX * body)

def _is_shooting_star(open_price: float, high: float, low: float, close: float) -> bool:
    """Checks if the given candle is a Shooting Star pattern."""
    body = abs(close - open_price)
    candle_range = high - low

    # Invalid candle if range is zero or body is too small to calculate ratios
    if candle_range == 0 or body < MIN_BODY_SIZE:
        return False

    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low

    # Conditions for a Shooting Star:
    # 1. Small body relative to total range
    # 2. Long upper shadow (at least 2x body)
    # 3. Very small or no lower shadow (less than 10% of body relative to body size)
    return (body < CANDLE_BODY_RATIO_MAX * candle_range and
            upper_shadow >= CANDLE_SHADOW_RATIO_MIN * body and
            lower_shadow < CANDLE_OPPOSITE_SHADOW_RATIO_MAX * body)

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Identifies Bollinger Band Reversal with Candlestick and Dual Confirmation signals.
    A Buy signal is generated when the price closes below the lower Bollinger Band,
    a Hammer candlestick forms, MFI indicates oversold conditions, and volume is
    above its average.
    A Sell signal is generated when the price closes above the upper Bollinger Band,
    a Shooting Star forms, MFI indicates overbought conditions, and volume is
    above its average.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract relevant data into numpy arrays for efficient calculation
        # Ensure enough data points are available for slicing
        opens = np.array([c.open_price for c in warm_candles])
        highs = np.array([c.high for c in warm_candles])
        lows = np.array([c.low for c in warm_candles])
        closes = np.array([c.close for c in warm_candles])
        volumes = np.array([c.volume for c in warm_candles])

        current_candle = warm_candles[-1]
        current_open = opens[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        current_close = closes[-1]
        current_volume = volumes[-1]

        # 1. Calculate Bollinger Bands for the last BB_PERIOD closes
        _, ubb, lbb = _calculate_bollinger_bands(closes, BB_PERIOD, BB_NUM_STD_DEV)
        if np.isnan(ubb) or np.isnan(lbb):
            # This should ideally not happen if MIN_CANDLES_REQUIRED is met, but good for robustness
            continue

        # 2. Calculate Money Flow Index (MFI) for the last MFI_PERIOD + 1 data points
        mfi = _calculate_mfi(highs, lows, closes, volumes, MFI_PERIOD)
        if np.isnan(mfi):
            continue

        # 3. Calculate Volume SMA for the last VOLUME_SMA_PERIOD volumes
        volume_sma = _calculate_sma(volumes, VOLUME_SMA_PERIOD)
        if np.isnan(volume_sma):
            continue

        # --- BUY Signal Conditions ---
        # IF current_close < LBB
        # AND current_candle is a Hammer
        # AND MFI < MFI_OVERSOLD_THRESHOLD
        # AND current_volume > SMA_Volume
        if (current_close < lbb and
            _is_hammer(current_open, current_high, current_low, current_close) and
            mfi < MFI_OVERSOLD_THRESHOLD and
            current_volume > volume_sma):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_close,
                rule_id=RULE_ID
            ))

        # --- SELL Signal Conditions ---
        # IF current_close > UBB
        # AND current_candle is a Shooting Star
        # AND MFI > MFI_OVERBOUGHT_THRESHOLD
        # AND current_volume > SMA_Volume
        elif (current_close > ubb and
              _is_shooting_star(current_open, current_high, current_low, current_close) and
              mfi > MFI_OVERBOUGHT_THRESHOLD and
              current_volume > volume_sma):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_close,
                rule_id=RULE_ID
            ))

    return signals