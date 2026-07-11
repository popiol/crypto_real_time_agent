from __future__ import annotations
import numpy as np
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal

# Rule ID for tracking
RULE_ID = "998fe0d5-947e-4747-ae89-7fb9acce3630"

# Constants from pseudocode
BB_PERIOD = 20
BB_DEV = 2
MFI_PERIOD = 14
MFI_OVERSOLD_THRESHOLD = 20
MFI_OVERBOUGHT_THRESHOLD = 80
VOLUME_SMA_PERIOD = 20
VOLUME_MULTIPLIER = 1.5

# Minimum number of candles required for calculations
MIN_CANDLES_REQUIRED = max(BB_PERIOD, MFI_PERIOD, VOLUME_SMA_PERIOD) + 1 # +1 for previous candle in MFI and engulfing

# Candlestick pattern parameters (relative to real body)
REAL_BODY_MIN_SIZE_FACTOR = 0.001 # Minimum real body size relative to candle range to avoid division by zero or tiny bodies
SHADOW_RATIO_LONG = 2.0
SHADOW_RATIO_SMALL = 0.5


def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    return np.convolve(data, np.ones(period)/period, mode='valid')

def _calculate_stddev(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Standard Deviation for a given period."""
    stddevs = np.zeros(len(data) - period + 1)
    for i in range(len(stddevs)):
        stddevs[i] = np.std(data[i:i+period])
    return stddevs

def _calculate_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Calculates Money Flow Index (MFI)."""
    if len(high) < period + 1:
        return np.array([])

    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_money_flow = np.zeros(len(typical_price) - period)
    negative_money_flow = np.zeros(len(typical_price) - period)

    for i in range(period, len(typical_price)):
        pmf_sum = 0.0
        nmf_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if typical_price[j] > typical_price[j-1]:
                pmf_sum += money_flow[j]
            elif typical_price[j] < typical_price[j-1]:
                nmf_sum += money_flow[j]
        positive_money_flow[i - period] = pmf_sum
        negative_money_flow[i - period] = nmf_sum

    money_ratio = np.where(negative_money_flow == 0, np.inf, positive_money_flow / negative_money_flow)
    mfi = np.where(money_ratio == np.inf, 100.0, np.where(money_ratio == 0, 0.0, 100 - (100 / (1 + money_ratio))))
    
    return mfi

def _is_hammer(o: float, h: float, l: float, c: float) -> bool:
    """Checks for a Hammer candlestick pattern."""
    real_body = abs(c - o)
    candle_range = h - l
    
    if candle_range == 0 or real_body / candle_range < REAL_BODY_MIN_SIZE_FACTOR:
        return False # Avoid tiny or zero real bodies

    # Body must be at the upper end of the candle range
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    # Conditions: small real body, long lower shadow, little or no upper shadow
    return (real_body < (candle_range * 0.3)) and \
           (lower_shadow >= SHADOW_RATIO_LONG * real_body) and \
           (upper_shadow < real_body * SHADOW_RATIO_SMALL) and \
           (c > o) # Must be a bullish hammer

def _is_shooting_star(o: float, h: float, l: float, c: float) -> bool:
    """Checks for a Shooting Star candlestick pattern."""
    real_body = abs(c - o)
    candle_range = h - l

    if candle_range == 0 or real_body / candle_range < REAL_BODY_MIN_SIZE_FACTOR:
        return False # Avoid tiny or zero real bodies

    # Body must be at the lower end of the candle range
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    
    # Conditions: small real body, long upper shadow, little or no lower shadow
    return (real_body < (candle_range * 0.3)) and \
           (upper_shadow >= SHADOW_RATIO_LONG * real_body) and \
           (lower_shadow < real_body * SHADOW_RATIO_SMALL) and \
           (c < o) # Must be a bearish shooting star

def _is_bullish_engulfing(prev_o: float, prev_c: float, curr_o: float, curr_c: float) -> bool:
    """Checks for a Bullish Engulfing pattern."""
    # Previous candle must be bearish, current candle must be bullish
    # Current candle's body must engulf the previous candle's body
    return (prev_c < prev_o) and \
           (curr_c > curr_o) and \
           (curr_o < prev_c) and \
           (curr_c > prev_o)

def _is_bearish_engulfing(prev_o: float, prev_c: float, curr_o: float, curr_c: float) -> bool:
    """Checks for a Bearish Engulfing pattern."""
    # Previous candle must be bullish, current candle must be bearish
    # Current candle's body must engulf the previous candle's body
    return (prev_c > prev_o) and \
           (curr_c < curr_o) and \
           (curr_o > prev_c) and \
           (curr_c < prev_o)

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Bollinger Band Reversal with MFI, Volume, and Candlestick Confirmation rule.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = sorted(pair_data.warm, key=lambda c: c.hour) # Ensure candles are in chronological order

        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract necessary data into numpy arrays
        closes = np.array([c.close for c in warm_candles])
        highs = np.array([c.high for c in warm_candles])
        lows = np.array([c.low for c in warm_candles])
        opens = np.array([c.open_price for c in warm_candles])
        volumes = np.array([c.volume for c in warm_candles])
        timestamps = [c.hour for c in warm_candles]

        # --- Calculate Indicators ---

        # Bollinger Bands
        sma_bb = _calculate_sma(closes, BB_PERIOD)
        stddev_bb = _calculate_stddev(closes, BB_PERIOD)
        
        # Adjust lengths for alignment: SMA and STDDEV will be shorter by BB_PERIOD - 1
        # We need the last value, so we take the last element of the calculated arrays.
        if len(sma_bb) == 0 or len(stddev_bb) == 0:
            continue
        
        upper_band = sma_bb[-1] + (stddev_bb[-1] * BB_DEV)
        lower_band = sma_bb[-1] - (stddev_bb[-1] * BB_DEV)

        # Money Flow Index
        mfi_values = _calculate_mfi(highs, lows, closes, volumes, MFI_PERIOD)
        if len(mfi_values) == 0:
            continue
        mfi_current = mfi_values[-1]

        # Volume SMA
        volume_sma_values = _calculate_sma(volumes, VOLUME_SMA_PERIOD)
        if len(volume_sma_values) == 0:
            continue
        volume_sma_current = volume_sma_values[-1]

        # --- Get current candle data ---
        current_candle = warm_candles[-1]
        prev_candle = warm_candles[-2] # Required for engulfing patterns

        current_close = current_candle.close
        current_open = current_candle.open_price
        current_high = current_candle.high
        current_low = current_candle.low
        current_volume = current_candle.volume
        
        prev_close = prev_candle.close
        prev_open = prev_candle.open_price

        # --- Check Candlestick Patterns ---
        is_hammer = _is_hammer(current_open, current_high, current_low, current_close)
        is_shooting_star = _is_shooting_star(current_open, current_high, current_low, current_close)
        is_bullish_engulfing = _is_bullish_engulfing(prev_open, prev_close, current_open, current_close)
        is_bearish_engulfing = _is_bearish_engulfing(prev_open, prev_close, current_open, current_close)

        # --- Evaluate Buy Signal ---
        buy_condition_bb_breach = (current_close < lower_band)
        buy_condition_mfi = (mfi_current < MFI_OVERSOLD_THRESHOLD)
        buy_condition_volume = (current_volume > (volume_sma_current * VOLUME_MULTIPLIER))
        buy_condition_candlestick = (is_hammer or is_bullish_engulfing)

        if (buy_condition_bb_breach and
                buy_condition_mfi and
                buy_condition_volume and
                buy_condition_candlestick):
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_close,
                rule_id=RULE_ID
            ))

        # --- Evaluate Sell Signal ---
        sell_condition_bb_breach = (current_close > upper_band)
        sell_condition_mfi = (mfi_current > MFI_OVERBOUGHT_THRESHOLD)
        sell_condition_volume = (current_volume > (volume_sma_current * VOLUME_MULTIPLIER))
        sell_condition_candlestick = (is_shooting_star or is_bearish_engulfing)

        if (sell_condition_bb_breach and
                sell_condition_mfi and
                sell_condition_volume and
                sell_condition_candlestick):
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_close,
                rule_id=RULE_ID
            ))

    return signals