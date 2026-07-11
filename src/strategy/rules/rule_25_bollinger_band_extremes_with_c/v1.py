from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule Configuration ---
RULE_ID = "ead1e5c6-13c6-498d-a1d8-35baa9f8b8ab"

# Indicator Parameters
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70
VOLUME_MA_PERIOD = 20
VOLUME_SURGE_MULTIPLIER = 1.5

# Minimum required candles for calculations
# RSI(N) needs N+1 candles to calculate the first N price changes.
# BB(N) and VMA(N) need N candles.
# So, min candles is max(BB_PERIOD, RSI_PERIOD + 1, VOLUME_MA_PERIOD).
MIN_CANDLES = max(BB_PERIOD, RSI_PERIOD + 1, VOLUME_MA_PERIOD)

# --- Helper Functions for Indicators ---

def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < period:
        return np.array([])
    return np.convolve(data, np.ones(period) / period, mode='valid')

def _calculate_std_dev(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Rolling Standard Deviation."""
    if len(data) < period:
        return np.array([])
    std_devs = np.zeros(len(data) - period + 1)
    for i in range(len(std_devs)):
        std_devs[i] = np.std(data[i:i + period])
    return std_devs

def _calculate_bollinger_bands(closes: np.ndarray, period: int, std_dev_multiplier: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates Bollinger Bands (Middle, Upper, Lower)."""
    if len(closes) < period:
        return np.array([]), np.array([]), np.array([])

    middle_band = _calculate_sma(closes, period)
    std_dev = _calculate_std_dev(closes, period)

    # Ensure std_dev and middle_band are of compatible length for element-wise operation
    # std_dev and middle_band from _calculate_sma/_calculate_std_dev will have same length
    # if called with the same `data` and `period` and `mode='valid'`
    upper_band = middle_band + (std_dev * std_dev_multiplier)
    lower_band = middle_band - (std_dev * std_dev_multiplier)

    return middle_band, upper_band, lower_band

def _calculate_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Calculates Relative Strength Index (RSI)."""
    if len(closes) < period + 1:
        return np.array([])

    diffs = np.diff(closes)
    gains = np.maximum(0, diffs)
    losses = np.maximum(0, -diffs)

    avg_gain = np.zeros_like(gains)
    avg_loss = np.zeros_like(losses)

    # Initial average gain/loss over the first 'period' diffs
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    # Smoothed average gain/loss using Wilder's smoothing method
    for i in range(period, len(diffs)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    # Calculate Relative Strength (RS)
    # Handle division by zero for avg_loss
    rs = np.divide(avg_gain[period - 1:], avg_loss[period - 1:],
                   out=np.full_like(avg_gain[period - 1:], np.nan),
                   where=avg_loss[period - 1:] != 0)

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _is_hammer(candle: WarmCandle) -> bool:
    """Checks for Hammer candlestick pattern based on pseudocode criteria."""
    total_range = candle.high - candle.low
    # Degenerate candle (High == Low) or invalid range
    if total_range <= 1e-6:
        return False

    body_range = abs(candle.open_price - candle.close)
    upper_shadow = candle.high - max(candle.open_price, candle.close)
    lower_shadow = min(candle.open_price, candle.close) - candle.low

    # Criteria from pseudocode:
    # 1. Current_Candle.Open > Current_Candle.Close (Bearish body)
    # 2. (Current_Candle.Open - Current_Candle.Close) < 0.3 * (Current_Candle.High - Current_Candle.Low) (Small body)
    # 3. (Current_Candle.Close - Current_Candle.Low) > 2 * (Current_Candle.Open - Current_Candle.Close) (Long lower shadow)
    # 4. (Current_Candle.High - Current_Candle.Open) < 0.1 * (Current_Candle.High - Current_Candle.Low) (Small upper shadow)

    return (candle.open_price > candle.close and
            body_range < 0.3 * total_range and
            lower_shadow > 2 * body_range and
            upper_shadow < 0.1 * total_range)

def _is_shooting_star(candle: WarmCandle) -> bool:
    """Checks for Shooting Star candlestick pattern based on pseudocode criteria."""
    total_range = candle.high - candle.low
    # Degenerate candle (High == Low) or invalid range
    if total_range <= 1e-6:
        return False

    body_range = abs(candle.open_price - candle.close)
    upper_shadow = candle.high - max(candle.open_price, candle.close)
    lower_shadow = min(candle.open_price, candle.close) - candle.low

    # Criteria from pseudocode:
    # 1. Current_Candle.Open < Current_Candle.Close (Bullish body)
    # 2. (Current_Candle.Close - Current_Candle.Open) < 0.3 * (Current_Candle.High - Current_Candle.Low) (Small body)
    # 3. (Current_Candle.High - Current_Candle.Open) > 2 * (Current_Candle.Close - Current_Candle.Open) (Long upper shadow)
    # 4. (Current_Candle.Close - Current_Candle.Low) < 0.1 * (Current_Candle.High - Current_Candle.Low) (Small lower shadow)

    return (candle.open_price < candle.close and
            body_range < 0.3 * total_range and
            upper_shadow > 2 * body_range and
            lower_shadow < 0.1 * total_range)

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Detects high-conviction mean-reversion opportunities using Bollinger Band extremes,
    candlestick reversal patterns (Hammer/Shooting Star), RSI, and volume confirmation.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough data for all indicator calculations
        if len(warm_candles) < MIN_CANDLES:
            continue

        # Extract relevant data as numpy arrays for efficient calculations
        closes = np.array([c.close for c in warm_candles])
        volumes = np.array([c.volume for c in warm_candles])

        current_candle = warm_candles[-1]
        current_close = closes[-1]
        current_volume = volumes[-1]

        # 1. Calculate Bollinger Bands
        _, bb_upper, bb_lower = _calculate_bollinger_bands(closes, BB_PERIOD, BB_STD_DEV_MULTIPLIER)
        if len(bb_lower) == 0: # Check if calculation was successful
            continue
        last_lower_band = bb_lower[-1]
        last_upper_band = bb_upper[-1]

        # 2. Calculate RSI
        rsi_values = _calculate_rsi(closes, RSI_PERIOD)
        if len(rsi_values) == 0 or np.isnan(rsi_values[-1]): # Check if calculation was successful and not NaN
            continue
        last_rsi = rsi_values[-1]

        # 3. Calculate Volume MA
        volume_ma_values = _calculate_sma(volumes, VOLUME_MA_PERIOD)
        if len(volume_ma_values) == 0: # Check if calculation was successful
            continue
        last_volume_ma = volume_ma_values[-1]

        # --- Check for Buy Signal ---
        # 1. Current_Price.Close < Bollinger_Bands.Lower_Band(20, 2)
        # 2. RSI(14) < 30 (Oversold condition)
        # 3. Candlestick_Pattern_Is_Hammer(Current_Candle)
        # 4. Current_Volume > 1.5 * Volume_MA(20) (Significant volume surge)
        buy_condition_1 = current_close < last_lower_band
        buy_condition_2 = last_rsi < RSI_OVERSOLD_THRESHOLD
        buy_condition_3 = _is_hammer(current_candle)
        buy_condition_4 = current_volume > (VOLUME_SURGE_MULTIPLIER * last_volume_ma)

        if buy_condition_1 and buy_condition_2 and buy_condition_3 and buy_condition_4:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_close,
                rule_id=RULE_ID,
                confidence=None # Optional: Could add a confidence score based on how strong conditions are
            ))

        # --- Check for Sell Signal ---
        # 1. Current_Price.Close > Bollinger_Bands.Upper_Band(20, 2)
        # 2. RSI(14) > 70 (Overbought condition)
        # 3. Candlestick_Pattern_Is_Shooting_Star(Current_Candle)
        # 4. Current_Volume > 1.5 * Volume_MA(20) (Significant volume surge)
        sell_condition_1 = current_close > last_upper_band
        sell_condition_2 = last_rsi > RSI_OVERBOUGHT_THRESHOLD
        sell_condition_3 = _is_shooting_star(current_candle)
        sell_condition_4 = current_volume > (VOLUME_SURGE_MULTIPLIER * last_volume_ma)

        if sell_condition_1 and sell_condition_2 and sell_condition_3 and sell_condition_4:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_close,
                rule_id=RULE_ID,
                confidence=None # Optional
            ))

    return signals