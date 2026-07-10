from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule Constants ---
BB_PERIOD = 20  # N-period for SMA and StdDev
BB_STD_DEV_FACTOR = 2.0  # K for StdDev multiplier
# Minimum candles needed: BB_PERIOD for calculation + 1 for the 'previous_candle' check + 1 for the 'current_candle' confirmation.
# The BBs are calculated using BB_PERIOD candles ending BEFORE the current candle.
# So, if we have N candles `c0, c1, ..., c(N-1)`, and `current_candle = c(N-1)`,
# `previous_candle = c(N-2)`.
# The BBs for `current_candle` are calculated using `c0, ..., c(N-2)`.
# This means we need at least `BB_PERIOD` candles for the BB calculation (c0 to c(BB_PERIOD-1)),
# and then `c(BB_PERIOD)` as `previous_candle`, and `c(BB_PERIOD+1)` as `current_candle`.
# So, total `BB_PERIOD + 2` candles are needed for the most robust interpretation.
# However, the pseudocode implies that the `previous_candle` is the one that breaches, and `current_candle` re-enters.
# This means the BBs should be calculated *up to* the `previous_candle`.
# Let's adjust: BBs are calculated using `candles[-(BB_PERIOD+1):-1]`. This window ends at `candles[-2]`.
# So `previous_candle` is `candles[-2]`, `current_candle` is `candles[-1]`.
# The BB calculation uses `BB_PERIOD` candles, the last of which is `candles[-2]`.
# Thus, we need `BB_PERIOD + 1` candles in total.
# E.g., if BB_PERIOD=20, we need 21 candles. `candles[0]` to `candles[20]`.
# `bb_closes` uses `candles[0]` to `candles[19]`.
# `previous_candle` is `candles[19]`. `current_candle` is `candles[20]`.
MIN_CANDLES = BB_PERIOD + 1

# --- Candlestick Pattern Helpers ---
def _is_bullish_candle(candle: WarmCandle) -> bool:
    """Checks if the candle closed higher than it opened."""
    return candle.close > candle.open_price

def _is_bearish_candle(candle: WarmCandle) -> bool:
    """Checks if the candle closed lower than it opened."""
    return candle.close < candle.open_price

def _candle_body_size(candle: WarmCandle) -> float:
    """Calculates the absolute size of the candle body."""
    return abs(candle.close - candle.open_price)

def _upper_shadow(candle: WarmCandle) -> float:
    """Calculates the length of the upper shadow."""
    return candle.high - max(candle.open_price, candle.close)

def _lower_shadow(candle: WarmCandle) -> float:
    """Calculates the length of the lower shadow."""
    return min(candle.open_price, candle.close) - candle.low

# Bullish Reversal Patterns
def _is_hammer(current_candle: WarmCandle) -> bool:
    """
    Detects a Hammer pattern.
    Small body near the top of the range, long lower shadow (at least twice the body),
    and a very small or no upper shadow.
    """
    body = _candle_body_size(current_candle)
    lower_shadow = _lower_shadow(current_candle)
    upper_shadow = _upper_shadow(current_candle)

    # Body must be non-zero to avoid division by zero and represent a meaningful candle.
    if body == 0:
        return False

    return (
        lower_shadow >= 2 * body
        and upper_shadow < body / 2  # Upper shadow should be very small
        and current_candle.close >= current_candle.open_price # Often bullish or neutral body
    )

def _is_bullish_engulfing(current_candle: WarmCandle, prev_candle: WarmCandle) -> bool:
    """
    Detects a Bullish Engulfing pattern.
    A small bearish candle followed by a large bullish candle whose body completely
    engulfs the body of the previous candle.
    """
    return (
        _is_bearish_candle(prev_candle)
        and _is_bullish_candle(current_candle)
        and current_candle.close > prev_candle.open_price
        and current_candle.open_price < prev_candle.close
    )

def _is_piercing_line(current_candle: WarmCandle, prev_candle: WarmCandle) -> bool:
    """
    Detects a Piercing Line pattern.
    A long bearish candle followed by a bullish candle that opens below the previous low
    and closes above the midpoint of the previous candle's body, but not above its open.
    """
    prev_midpoint = (prev_candle.open_price + prev_candle.close) / 2
    return (
        _is_bearish_candle(prev_candle)
        and _is_bullish_candle(current_candle)
        and current_candle.open_price < prev_candle.low
        and current_candle.close > prev_midpoint
        and current_candle.close < prev_candle.open_price # Should not close above previous open (then it's engulfing)
    )

# Bearish Reversal Patterns
def _is_shooting_star(current_candle: WarmCandle) -> bool:
    """
    Detects a Shooting Star pattern.
    Small body near the bottom of the range, long upper shadow (at least twice the body),
    and a very small or no lower shadow.
    """
    body = _candle_body_size(current_candle)
    lower_shadow = _lower_shadow(current_candle)
    upper_shadow = _upper_shadow(current_candle)

    if body == 0:
        return False

    return (
        upper_shadow >= 2 * body
        and lower_shadow < body / 2  # Lower shadow should be very small
        and current_candle.close <= current_candle.open_price # Often bearish or neutral body
    )

def _is_bearish_engulfing(current_candle: WarmCandle, prev_candle: WarmCandle) -> bool:
    """
    Detects a Bearish Engulfing pattern.
    A small bullish candle followed by a large bearish candle whose body completely
    engulfs the body of the previous candle.
    """
    return (
        _is_bullish_candle(prev_candle)
        and _is_bearish_candle(current_candle)
        and current_candle.open_price > prev_candle.close
        and current_candle.close < prev_candle.open_price
    )

def _is_dark_cloud_cover(current_candle: WarmCandle, prev_candle: WarmCandle) -> bool:
    """
    Detects a Dark Cloud Cover pattern.
    A long bullish candle followed by a bearish candle that opens above the previous high
    and closes below the midpoint of the previous candle's body, but not below its close.
    """
    prev_midpoint = (prev_candle.open_price + prev_candle.close) / 2
    return (
        _is_bullish_candle(prev_candle)
        and _is_bearish_candle(current_candle)
        and current_candle.open_price > prev_candle.high
        and current_candle.close < prev_midpoint
        and current_candle.close > prev_candle.close # Should not close below previous close (then it's engulfing)
    )

def is_bullish_reversal_pattern(current_candle: WarmCandle, prev_candle: WarmCandle) -> bool:
    """Checks for any of the specified bullish reversal candlestick patterns."""
    return (
        _is_hammer(current_candle)
        or _is_bullish_engulfing(current_candle, prev_candle)
        or _is_piercing_line(current_candle, prev_candle)
    )

def is_bearish_reversal_pattern(current_candle: WarmCandle, prev_candle: WarmCandle) -> bool:
    """Checks for any of the specified bearish reversal candlestick patterns."""
    return (
        _is_shooting_star(current_candle)
        or _is_bearish_engulfing(current_candle, prev_candle)
        or _is_dark_cloud_cover(current_candle, prev_candle)
    )

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Bollinger Band Reversal with Candlestick Confirmation' trading rule.

    This rule detects potential mean-reversion reversals when the price initially breaches
    a Bollinger Band (lower for buy, upper for sell) and subsequently closes back inside
    the band, followed by a bullish (for buy) or bearish (for sell) candlestick pattern.

    Args:
        data: MarketData object containing tick and candle data for various pairs.

    Returns:
        A list of BuySignal or SellSignal objects if the rule conditions are met.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # Ensure enough candles for Bollinger Band calculation and pattern detection.
        # We need BB_PERIOD candles to calculate the bands, then 2 more for the previous and current candle.
        # The BB calculation uses `candles[-(BB_PERIOD+1):-1]`, which are `BB_PERIOD` candles.
        # `previous_candle` is `candles[-2]`, `current_candle` is `candles[-1]`.
        if len(candles) < MIN_CANDLES:
            continue

        # Extract closing prices for Bollinger Band calculation.
        # The BBs for the `current_candle` are calculated based on the `BB_PERIOD` candles
        # immediately preceding the `current_candle` (i.e., ending at `previous_candle`).
        bb_closes = np.array([c.close for c in candles[-(BB_PERIOD + 1):-1]])

        # If for some reason (e.g., edge case with very few candles near MIN_CANDLES)
        # bb_closes doesn't have enough data, skip.
        if len(bb_closes) < BB_PERIOD:
            continue

        middle_band = np.mean(bb_closes)
        std_dev = np.std(bb_closes)
        upper_band = middle_band + BB_STD_DEV_FACTOR * std_dev
        lower_band = middle_band - BB_STD_DEV_FACTOR * std_dev

        current_candle = candles[-1]
        previous_candle = candles[-2]

        # --- Buy Signal Logic ---
        # 1. The previous candle's low was below the Lower Bollinger Band (price dipped below).
        # 2. The current close price is now *inside* the Bollinger Bands (re-entry).
        # 3. A bullish reversal candlestick pattern is observed on the current candle.
        if (
            previous_candle.low < lower_band
            and current_candle.close > lower_band
            and current_candle.close < upper_band
            and is_bullish_reversal_pattern(current_candle, previous_candle)
        ):
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id="BollingerBandReversalWithCandlestickConfirmation"
            ))

        # --- Sell Signal Logic ---
        # 1. The previous candle's high was above the Upper Bollinger Band (price peaked above).
        # 2. The current close price is now *inside* the Bollinger Bands (re-entry).
        # 3. A bearish reversal candlestick pattern is observed on the current candle.
        elif ( # Use elif to prioritize a single signal per candle per pair if conditions overlap (unlikely here)
            previous_candle.high > upper_band
            and current_candle.close > lower_band
            and current_candle.close < upper_band
            and is_bearish_reversal_pattern(current_candle, previous_candle)
        ):
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id="BollingerBandReversalWithCandlestickConfirmation"
            ))

    return signals