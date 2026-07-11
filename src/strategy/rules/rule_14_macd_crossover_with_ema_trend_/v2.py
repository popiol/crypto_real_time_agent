from __future__ import annotations
from datetime import datetime
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal

# --- Configuration ---
# Periods for MACD components
SHORT_EMA_PERIOD = 12
LONG_EMA_PERIOD = 26
SIGNAL_EMA_PERIOD = 9
# Period for the long-term trend confirmation EMA
TREND_EMA_PERIOD = 200
# Number of periods to check EMA slope
EMA_SLOPE_LOOKBACK = 5

# Minimum data points required for all calculations and crossover detection.
# This must be sufficient for:
# 1. All base EMAs (SHORT_EMA, LONG_EMA, TREND_EMA) to be calculated.
# 2. MACD signal line EMA to be calculated.
# 3. Accessing current and previous values for MACD crossover (at least 2 points).
# 4. Accessing current and EMA_SLOPE_LOOKBACK-ago values for EMA slope (at least EMA_SLOPE_LOOKBACK points).
# If `_calculate_ema` returns an array of length `len(prices)` when `len(prices) >= period`,
# then `MIN_CANDLES_REQUIRED` is the maximum of all periods and lookbacks.
MIN_CANDLES_REQUIRED = max(
    SHORT_EMA_PERIOD,
    LONG_EMA_PERIOD,
    SIGNAL_EMA_PERIOD,
    TREND_EMA_PERIOD,
    EMA_SLOPE_LOOKBACK
)

# Unique identifier for this trading rule
RULE_ID = "036a4c0d-5899-4574-a1eb-cc1c589c2fd1"

def _calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates the Exponential Moving Average (EMA) for a given set of prices.
    The EMA is calculated iteratively, with the first EMA value initialized to the first price.
    Returns an array of the same length as prices. If insufficient data, returns an empty array.
    """
    if len(prices) < period:
        # Not enough data for the specified period
        return np.array([])
    
    ema = np.zeros(len(prices), dtype=np.float64)
    multiplier = 2 / (period + 1)

    # Initialize the first EMA value with the first price.
    # For sufficiently long series, the latest EMA values will be stable.
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
    return ema

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Revise MACD-EMA Trend Confirmation with EMA Slope' trading rule.

    This rule detects shifts in short-term momentum using the MACD crossover,
    but filters these signals based on the direction (slope) of a longer-term
    Exponential Moving Average (EMA) to confirm the underlying trend.

    A Buy signal is emitted when the MACD line crosses above its signal line
    while the long-term EMA is upward-sloping.
    A Sell signal is emitted when the MACD line crosses below its signal line
    while the long-term EMA is downward-sloping.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # Ensure we have enough historical data to calculate all required EMAs
        # and detect crossovers/slopes.
        if len(candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract closing prices from the candles
        close_prices = np.array([c.close for c in candles], dtype=np.float64)

        # 1. Calculate MACD Line (12-period EMA - 26-period EMA)
        ema_short = _calculate_ema(close_prices, SHORT_EMA_PERIOD)
        ema_long = _calculate_ema(close_prices, LONG_EMA_PERIOD)

        # If any base EMA calculation fails due to insufficient data (should be caught by MIN_CANDLES_REQUIRED)
        if len(ema_short) == 0 or len(ema_long) == 0:
            continue 

        macd_line = ema_short - ema_long

        # 2. Calculate MACD Signal line (9-period EMA of the MACD line)
        macd_signal_line = _calculate_ema(macd_line, SIGNAL_EMA_PERIOD)

        if len(macd_signal_line) == 0:
            continue 

        # 3. Calculate Long-term Exponential Moving Average (e.g., 200-period EMA)
        long_term_ema = _calculate_ema(close_prices, TREND_EMA_PERIOD)

        if len(long_term_ema) == 0:
            continue 

        # At this point, all indicator arrays should have len(close_prices) elements,
        # which is guaranteed to be at least MIN_CANDLES_REQUIRED.
        # This ensures we can safely access elements like [-1], [-2], and [-EMA_SLOPE_LOOKBACK].

        # Get current and previous values for indicators for crossover analysis
        current_close = close_prices[-1]
        
        current_macd = macd_line[-1]
        prev_macd = macd_line[-2]

        current_macd_signal = macd_signal_line[-1]
        prev_macd_signal = macd_signal_line[-2]

        # Get current and past long-term EMA values for slope analysis
        current_long_term_ema = long_term_ema[-1]
        # We need to ensure that EMA_SLOPE_LOOKBACK does not exceed the array length.
        # MIN_CANDLES_REQUIRED ensures long_term_ema has at least EMA_SLOPE_LOOKBACK elements.
        past_long_term_ema_for_slope = long_term_ema[-EMA_SLOPE_LOOKBACK]

        # 4. Determine EMA Slope
        ema_slope_up = current_long_term_ema > past_long_term_ema_for_slope
        ema_slope_down = current_long_term_ema < past_long_term_ema_for_slope

        # 5. Buy Signal: MACD line crosses above Signal line AND Long-term EMA is upward-sloping.
        macd_cross_up = (prev_macd < prev_macd_signal) and (current_macd > current_macd_signal)
        
        if macd_cross_up and ema_slope_up:
            signals.append(BuySignal(
                pair=pair,
                timestamp=candles[-1].hour, # Use the timestamp of the latest candle
                price=current_close,
                rule_id=RULE_ID,
            ))

        # 6. Sell Signal: MACD line crosses below Signal line AND Long-term EMA is downward-sloping.
        macd_cross_down = (prev_macd > prev_macd_signal) and (current_macd < current_macd_signal)
        
        if macd_cross_down and ema_slope_down:
            signals.append(SellSignal(
                pair=pair,
                timestamp=candles[-1].hour, # Use the timestamp of the latest candle
                price=current_close,
                rule_id=RULE_ID,
            ))

    return signals