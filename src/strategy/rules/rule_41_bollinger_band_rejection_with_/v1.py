from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle
from datetime import datetime

# Parameters
BB_PERIOD = 20
BB_STD_DEV = 2
MFI_PERIOD = 14
MFI_OVERSOLD_THRESHOLD = 20
MFI_OVERBOUGHT_THRESHOLD = 80
VOLUME_SMA_PERIOD = 20
BB_WIDTH_SMA_PERIOD = 20
CANDLE_BODY_RATIO = 0.3 # e.g., body must be at least 30% of total range for engulfing
WICK_RATIO = 2.0 # e.g., wick must be at least 2x body for hammer/shooting star

# --- Helper Functions for Indicators ---

def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average, padding with NaN to match original data length."""
    if len(data) < period:
        return np.full_like(data, np.nan)
    
    weights = np.ones(period) / period
    sma_vals = np.convolve(data, weights, mode='valid')
    
    # Pad with NaN at the beginning to match original data length
    return np.concatenate((np.full(period - 1, np.nan), sma_vals))

def _std_dev(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Standard Deviation, padding with NaN to match original data length."""
    if len(data) < period:
        return np.full_like(data, np.nan)
    
    std_devs = np.array([np.std(data[i-period+1:i+1]) for i in range(period-1, len(data))])
    
    # Pad with NaN at the beginning to match original data length
    return np.concatenate((np.full(period - 1, np.nan), std_devs))

def _money_flow_index(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Calculates Money Flow Index (MFI), padding with NaN to match original data length."""
    if len(high) < period:
        return np.full_like(high, np.nan)

    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_money_flow = np.zeros_like(typical_price)
    negative_money_flow = np.zeros_like(typical_price)

    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_money_flow[i] = money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_money_flow[i] = money_flow[i]

    mfi_values = np.full_like(typical_price, np.nan)

    for i in range(period - 1, len(typical_price)):
        period_pos_mf = np.sum(positive_money_flow[i - period + 1 : i + 1])
        period_neg_mf = np.sum(negative_money_flow[i - period + 1 : i + 1])

        if period_neg_mf == 0:
            mfi_values[i] = 100.0 # Avoid division by zero, strong bullish
        else:
            money_ratio = period_pos_mf / period_neg_mf
            mfi_values[i] = 100 - (100 / (1 + money_ratio))
    
    return mfi_values

# --- Helper Functions for Candlestick Patterns ---

def _get_candle_properties(candle: WarmCandle):
    """Extracts properties for candlestick analysis."""
    open_p, high_p, low_p, close_p = candle.open_price, candle.high, candle.low, candle.close
    body = abs(close_p - open_p)
    total_range = high_p - low_p
    is_bullish = close_p > open_p
    is_bearish = close_p < open_p
    upper_wick = high_p - max(open_p, close_p)
    lower_wick = min(open_p, close_p) - low_p
    return open_p, high_p, low_p, close_p, body, total_range, is_bullish, is_bearish, upper_wick, lower_wick

def _is_bullish_engulfing(current_candle: WarmCandle, previous_candle: WarmCandle) -> bool:
    """Detects a Bullish Engulfing pattern."""
    if not previous_candle or not current_candle:
        return False

    prev_open, _, _, prev_close, prev_body, _, _, prev_is_bearish, _, _ = _get_candle_properties(previous_candle)
    curr_open, _, _, curr_close, curr_body, curr_total_range, curr_is_bullish, _, _, _ = _get_candle_properties(current_candle)

    # Current candle must be bullish, previous must be bearish
    # Current body must engulf previous body
    # Current close must be higher than previous open
    # Current open must be lower than previous close
    return (curr_is_bullish and prev_is_bearish and
            curr_body > prev_body and
            curr_close > prev_open and
            curr_open < prev_close and
            curr_total_range > 0 and # Avoid division by zero in ratio calculation
            curr_body / curr_total_range >= CANDLE_BODY_RAT) # Ensure current candle has a significant body

def _is_hammer(current_candle: WarmCandle) -> bool:
    """Detects a Hammer pattern."""
    _, _, _, _, body, total_range, _, _, upper_wick, lower_wick = _get_candle_properties(current_candle)

    if total_range == 0 or body == 0: # Avoid division by zero
        return False

    body_ratio_to_range = body / total_range
    lower_wick_ratio_to_body = lower_wick / body
    upper_wick_ratio_to_body = upper_wick / body

    # Hammer conditions:
    # 1. Small body (e.g., body < CANDLE_BODY_RAT of total range)
    # 2. Long lower wick (at least WICK_RATIO * body)
    # 3. Little/no upper wick (e.g., upper wick < body)
    # 4. Body is in the upper half of the candle's range.
    
    is_small_body = body_ratio_to_range < CANDLE_BODY_RAT
    is_long_lower_wick = lower_wick_ratio_to_body >= WICK_RATIO
    is_small_upper_wick = upper_wick_ratio_to_body < 1.0 

    body_midpoint = (current_candle.open_price + current_candle.close) / 2
    is_body_in_upper_half = (body_midpoint - current_candle.low) / total_range >= 0.5

    return is_small_body and is_long_lower_wick and is_small_upper_wick and is_body_in_upper_half

def _is_bearish_engulfing(current_candle: WarmCandle, previous_candle: WarmCandle) -> bool:
    """Detects a Bearish Engulfing pattern."""
    if not previous_candle or not current_candle:
        return False

    prev_open, _, _, prev_close, prev_body, _, prev_is_bullish, _, _, _ = _get_candle_properties(previous_candle)
    curr_open, _, _, curr_close, curr_body, curr_total_range, _, curr_is_bearish, _, _ = _get_candle_properties(current_candle)

    # Current candle must be bearish, previous must be bullish
    # Current body must engulf previous body
    # Current close must be lower than previous open
    # Current open must be higher than previous close
    return (curr_is_bearish and prev_is_bullish and
            curr_body > prev_body and
            curr_close < prev_open and
            curr_open > prev_close and
            curr_total_range > 0 and # Avoid division by zero in ratio calculation
            curr_body / curr_total_range >= CANDLE_BODY_RAT) # Ensure current candle has a significant body

def _is_shooting_star(current_candle: WarmCandle) -> bool:
    """Detects a Shooting Star pattern."""
    _, _, _, _, body, total_range, _, _, upper_wick, lower_wick = _get_candle_properties(current_candle)

    if total_range == 0 or body == 0: # Avoid division by zero
        return False

    body_ratio_to_range = body / total_range
    upper_wick_ratio_to_body = upper_wick / body
    lower_wick_ratio_to_body = lower_wick / body

    # Shooting Star conditions:
    # 1. Small body (e.g., body < CANDLE_BODY_RAT of total range)
    # 2. Long upper wick (at least WICK_RATIO * body)
    # 3. Little/no lower wick (e.g., lower wick < body)
    # 4. Body is in the lower half of the candle's range.

    is_small_body = body_ratio_to_range < CANDLE_BODY_RAT
    is_long_upper_wick = upper_wick_ratio_to_body >= WICK_RATIO
    is_small_lower_wick = lower_wick_ratio_to_body < 1.0 
    
    body_midpoint = (current_candle.open_price + current_candle.close) / 2
    is_body_in_lower_half = (body_midpoint - current_candle.low) / total_range <= 0.5

    return is_small_body and is_long_upper_wick and is_small_lower_wick and is_body_in_lower_half


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    # Determine maximum lookback period needed for all indicators
    # We need `max_lookback` candles to get the first valid indicator value
    # and `max_lookback + 1` to get both current and previous values for momentum checks.
    max_lookback = max(BB_PERIOD, MFI_PERIOD, VOLUME_SMA_PERIOD, BB_WIDTH_SMA_PERIOD)
    min_candles_required = max_lookback + 1 

    for pair, pair_data in data.items():
        candles = pair_data.warm

        if len(candles) < min_candles_required:
            continue

        # Extract data into numpy arrays for efficient computation
        closes = np.array([c.close for c in candles])
        opens = np.array([c.open_price for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Calculate Indicators (all padded with NaN to match original length)
        # SMA for Bollinger Bands
        bb_sma = _sma(closes, BB_PERIOD)
        bb_std_dev = _std_dev(closes, BB_PERIOD)
        upper_bb = bb_sma + (bb_std_dev * BB_STD_DEV)
        lower_bb = bb_sma - (bb_std_dev * BB_STD_DEV)

        mfi = _money_flow_index(highs, lows, closes, volumes, MFI_PERIOD)
        volume_sma = _sma(volumes, VOLUME_SMA_PERIOD)

        # Bollinger Band Width and its SMA
        # Ensure that bb_width_raw is only calculated where bb_sma is not NaN to avoid division by zero/NaN issues
        bb_width_raw = np.full_like(bb_sma, np.nan)
        valid_indices = ~np.isnan(bb_sma)
        # Only calculate where SMA is not zero to avoid division by zero
        valid_indices = valid_indices & (bb_sma != 0)
        bb_width_raw[valid_indices] = (upper_bb[valid_indices] - lower_bb[valid_indices]) / bb_sma[valid_indices]
        bb_width_sma = _sma(bb_width_raw, BB_WIDTH_SMA_PERIOD)

        # Check if latest indicator values are valid (not NaN)
        # All indicators should be aligned, so checking the last element is sufficient.
        if np.isnan(lower_bb[-1]) or np.isnan(upper_bb[-1]) or \
           np.isnan(mfi[-1]) or np.isnan(mfi[-2]) or \
           np.isnan(volume_sma[-1]) or np.isnan(bb_width_raw[-1]) or np.isnan(bb_width_sma[-1]):
            continue # Not enough valid data for current candle and previous MFI

        # --- Signal Logic for the current (last) candle ---
        current_candle = candles[-1]
        previous_candle = candles[-2] # Needed for engulfing and MFI momentum

        # Get the latest indicator values
        current_lower_bb = lower_bb[-1]
        current_upper_bb = upper_bb[-1]
        current_mfi = mfi[-1]
        previous_mfi = mfi[-2] # For MFI momentum check
        current_volume_sma = volume_sma[-1]
        current_bb_width = bb_width_raw[-1] # Current BB Width
        current_bb_width_sma = bb_width_sma[-1] # SMA of BB Width
        current_volume = volumes[-1]
        
        # Buy Signal Logic
        is_bullish_candle_confirmation = _is_bullish_engulfing(current_candle, previous_candle) or _is_hammer(current_candle)

        if current_candle.low <= current_lower_bb: # Price touched or breached lower BB
            if is_bullish_candle_confirmation:
                # MFI oversold AND turning up
                if current_mfi < MFI_OVERSOLD_THRESHOLD and current_mfi > previous_mfi:
                    # Volume confirmation AND elevated volatility
                    if current_volume > current_volume_sma and current_bb_width > current_bb_width_sma:
                        signals.append(BuySignal(
                            pair=pair,
                            timestamp=current_candle.hour,
                            price=current_candle.close,
                            rule_id="45d1a978-5575-4128-97b8-6eb97416a99a",
                            confidence=0.855 # Based on the score provided
                        ))

        # Sell Signal Logic
        is_bearish_candle_confirmation = _is_bearish_engulfing(current_candle, previous_candle) or _is_shooting_star(current_candle)

        if current_candle.high >= current_upper_bb: # Price touched or breached upper BB
            if is_bearish_candle_confirmation:
                # MFI overbought AND turning down
                if current_mfi > MFI_OVERBOUGHT_THRESHOLD and current_mfi < previous_mfi:
                    # Volume confirmation AND elevated volatility
                    if current_volume > current_volume_sma and current_bb_width > current_bb_width_sma:
                        signals.append(SellSignal(
                            pair=pair,
                            timestamp=current_candle.hour,
                            price=current_candle.close,
                            rule_id="45d1a978-5575-4128-97b8-6eb97416a99a",
                            confidence=0.855 # Based on the score provided
                        ))

    return signals