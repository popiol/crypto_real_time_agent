from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Constants ---
BB_PERIOD = 20
BB_STD_DEV = 2.0
# Minimum candles required: BB_PERIOD for the first Bollinger Band calculation,
# plus one more candle (Candle B) for the reversal pattern confirmation.
MIN_CANDLES_FOR_RULE = BB_PERIOD + 1

# --- Helper Functions for Candlestick Patterns ---

def _is_bullish_candle(candle: WarmCandle) -> bool:
    """Checks if a candle is bullish (close > open)."""
    return candle.close > candle.open_price

def _is_bearish_candle(candle: WarmCandle) -> bool:
    """Checks if a candle is bearish (close < open)."""
    return candle.close < candle.open_price

def _get_body_size(candle: WarmCandle) -> float:
    """Calculates the absolute size of the candle body."""
    return abs(candle.close - candle.open_price)

def _get_upper_shadow(candle: WarmCandle) -> float:
    """Calculates the length of the upper shadow."""
    return candle.high - max(candle.open_price, candle.close)

def _get_lower_shadow(candle: WarmCandle) -> float:
    """Calculates the length of the lower shadow."""
    return min(candle.open_price, candle.close) - candle.low

def _is_hammer(candle: WarmCandle) -> bool:
    """
    Detects a Hammer candlestick pattern.
    A hammer is a bullish reversal pattern.
    """
    body = _get_body_size(candle)
    lower_shadow = _get_lower_shadow(candle)
    upper_shadow = _get_upper_shadow(candle)

    # Must be a bullish candle for a strong hammer signal
    if not _is_bullish_candle(candle):
        return False

    total_range = candle.high - candle.low
    if total_range == 0:  # Avoid division by zero for flat candles
        return False

    # 1. Small body relative to total range (e.g., body < 35% of total range)
    if body / total_range > 0.35:
        return False

    # 2. Long lower shadow (at least 2x body)
    if lower_shadow < 2 * body:
        return False

    # 3. Very small or no upper shadow (e.g., less than 20% of body size)
    if upper_shadow > 0.2 * body:
        return False

    return True

def _is_shooting_star(candle: WarmCandle) -> bool:
    """
    Detects a Shooting Star candlestick pattern.
    A shooting star is a bearish reversal pattern.
    """
    body = _get_body_size(candle)
    upper_shadow = _get_upper_shadow(candle)
    lower_shadow = _get_lower_shadow(candle)

    # Must be a bearish candle for a strong shooting star signal
    if not _is_bearish_candle(candle):
        return False

    total_range = candle.high - candle.low
    if total_range == 0:  # Avoid division by zero for flat candles
        return False

    # 1. Small body relative to total range (e.g., body < 35% of total range)
    if body / total_range > 0.35:
        return False

    # 2. Long upper shadow (at least 2x body)
    if upper_shadow < 2 * body:
        return False

    # 3. Very small or no lower shadow (e.g., less than 20% of body size)
    if lower_shadow > 0.2 * body:
        return False

    return True

def _is_bullish_engulfing(prev_candle: WarmCandle, curr_candle: WarmCandle) -> bool:
    """
    Detects a Bullish Engulfing pattern.
    Requires the current bullish candle's body to completely engulf the previous bearish candle's body.
    """
    # 1. Previous candle must be bearish
    if not _is_bearish_candle(prev_candle):
        return False

    # 2. Current candle must be bullish
    if not _is_bullish_candle(curr_candle):
        return False

    # 3. Current candle's body must completely engulf previous candle's body
    # Current open is below previous close AND Current close is above previous open
    if not (curr_candle.open_price < prev_candle.close and curr_candle.close > prev_candle.open_price):
        return False

    return True

def _is_bearish_engulfing(prev_candle: WarmCandle, curr_candle: WarmCandle) -> bool:
    """
    Detects a Bearish Engulfing pattern.
    Requires the current bearish candle's body to completely engulf the previous bullish candle's body.
    """
    # 1. Previous candle must be bullish
    if not _is_bullish_candle(prev_candle):
        return False

    # 2. Current candle must be bearish
    if not _is_bearish_candle(curr_candle):
        return False

    # 3. Current candle's body must completely engulf previous candle's body
    # Current open is above previous close AND Current close is below previous open
    if not (curr_candle.open_price > prev_candle.close and curr_candle.close < prev_candle.open_price):
        return False

    return True

def _is_bullish_reversal_pattern(prev_candle: WarmCandle, curr_candle: WarmCandle) -> bool:
    """Checks if Candle B forms any recognized bullish reversal pattern."""
    return _is_hammer(curr_candle) or _is_bullish_engulfing(prev_candle, curr_candle)

def _is_bearish_reversal_pattern(prev_candle: WarmCandle, curr_candle: WarmCandle) -> bool:
    """Checks if Candle B forms any recognized bearish reversal pattern."""
    return _is_shooting_star(curr_candle) or _is_bearish_engulfing(prev_candle, curr_candle)

# --- Bollinger Band Calculation ---
def _calculate_bollinger_bands_for_window(
    candles: list[WarmCandle], period: int, std_dev: float
) -> tuple[float, float, float] | tuple[None, None, None]:
    """
    Calculates Bollinger Bands for a specific window of candles.
    This function expects exactly `period` number of candles in the input list.
    Returns (SMA, Upper Band, Lower Band) or (None, None, None) if data is insufficient.
    """
    if len(candles) != period:
        return None, None, None

    closes = np.array([c.close for c in candles])
    
    sma = np.mean(closes)
    std = np.std(closes)
    
    upper_band = sma + std * std_dev
    lower_band = sma - std * std_dev
    
    return sma, upper_band, lower_band


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Bollinger Band Reversal with Candlestick Confirmation rule.

    Emits a Buy signal when:
    1. Candle A closes below the Lower Bollinger Band.
    2. Candle B (immediately subsequent) forms a bullish reversal pattern (Hammer or Bullish Engulfing).

    Emits a Sell signal when:
    1. Candle A closes above the Upper Bollinger Band.
    2. Candle B (immediately subsequent) forms a bearish reversal pattern (Shooting Star or Bearish Engulfing).
    """
    signals: list[BuySignal | SellSignal] = []
    rule_id = "dbb7e41b-4f5a-45f8-839b-5c67a0a5046d"

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough candles for Bollinger Band calculation and subsequent pattern detection
        if len(warm_candles) < MIN_CANDLES_FOR_RULE:
            continue

        # Iterate through candles starting from the point where BB can be calculated for Candle A.
        # 'i' will be the index of Candle B.
        # 'i - 1' will be the index of Candle A.
        # The window for BB calculation for Candle A is warm_candles[i - BB_PERIOD : i].
        for i in range(BB_PERIOD, len(warm_candles)):
            # Define Candle A and Candle B
            candle_A = warm_candles[i - 1] # The candle that potentially breaches the band
            candle_B = warm_candles[i]     # The confirmation candle

            # Get the window of `BB_PERIOD` candles ending at Candle A for BB calculation
            window_for_bb = warm_candles[i - BB_PERIOD : i]
            
            sma_val, upper_band_val, lower_band_val = _calculate_bollinger_bands_for_window(
                window_for_bb, BB_PERIOD, BB_STD_DEV
            )

            if sma_val is None: # Should not happen if loop range is correct, but as safeguard
                continue

            # --- Buy Signal Logic ---
            # Condition 1: Candle A closes below the Lower Bollinger Band
            if candle_A.close < lower_band_val:
                # Condition 2: Candle B forms a bullish reversal pattern
                if _is_bullish_reversal_pattern(candle_A, candle_B):
                    signals.append(BuySignal(
                        pair=pair,
                        timestamp=candle_B.hour, # Signal at the close of Candle B
                        price=candle_B.close,
                        rule_id=rule_id,
                        confidence=None # Not specified in the rule idea
                    ))

            # --- Sell Signal Logic ---
            # Condition 1: Candle A closes above the Upper Bollinger Band
            if candle_A.close > upper_band_val:
                # Condition 2: Candle B forms a bearish reversal pattern
                if _is_bearish_reversal_pattern(candle_A, candle_B):
                    signals.append(SellSignal(
                        pair=pair,
                        timestamp=candle_B.hour, # Signal at the close of Candle B
                        price=candle_B.close,
                        rule_id=rule_id,
                        confidence=None # Not specified in the rule idea
                    ))
    return signals