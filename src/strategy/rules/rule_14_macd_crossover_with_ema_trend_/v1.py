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

# Minimum data points required for all calculations and crossover detection.
# We need `TREND_EMA_PERIOD` candles for the longest EMA, and then at least one previous
# candle to detect a crossover reliably (current vs. previous state).
MIN_CANDLES_REQUIRED = max(LONG_EMA_PERIOD, TREND_EMA_PERIOD) + 1 

# Unique identifier for this trading rule
RULE_ID = "7811c4a7-3ca2-414e-b49a-d9515f765806"

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
    Implements the 'MACD Crossover with EMA Trend Confirmation' trading rule.

    This rule detects shifts in short-term momentum using the MACD crossover,
    but filters these signals based on the direction of a longer-term Exponential
    Moving Average (EMA) to confirm the underlying trend.

    A Buy signal is emitted when the MACD line crosses above its signal line
    while the price is above the long-term EMA.
    A Sell signal is emitted when the MACD line crosses below its signal line
    while the price is below the long-term EMA.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # Ensure we have enough historical data to calculate all required EMAs
        # and detect crossovers (needs current and previous values).
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

        # We need at least two points for crossover detection (current and previous values).
        # This check is a safeguard, as MIN_CANDLES_REQUIRED should ensure this.
        if len(macd_line) < 2 or len(macd_signal_line) < 2 or len(long_term_ema) < 2:
            continue

        # Get current and previous values for indicators and price for analysis
        current_close = close_prices[-1]
        
        current_macd = macd_line[-1]
        prev_macd = macd_line[-2]

        current_macd_signal = macd_signal_line[-1]
        prev_macd_signal = macd_signal_line[-2]

        current_long_term_ema = long_term_ema[-1]

        # 4. Buy Signal: If MACD line crosses above Signal line AND Current Closing Price is above 200-period EMA.
        # Crossover detection: previous MACD was below signal, current MACD is above signal
        macd_cross_up = (prev_macd < prev_macd_signal) and (current_macd > current_macd_signal)
        price_above_ema = current_close > current_long_term_ema

        if macd_cross_up and price_above_ema:
            signals.append(BuySignal(
                pair=pair,
                timestamp=candles[-1].hour, # Use the timestamp of the latest candle
                price=current_close,
                rule_id=RULE_ID,
            ))

        # 5. Sell Signal: If MACD line crosses below Signal line AND Current Closing Price is below 200-period EMA.
        # Crossover detection: previous MACD was above signal, current MACD is below signal
        macd_cross_down = (prev_macd > prev_macd_signal) and (current_macd < current_macd_signal)
        price_below_ema = current_close < current_long_term_ema

        if macd_cross_down and price_below_ema:
            signals.append(SellSignal(
                pair=pair,
                timestamp=candles[-1].hour, # Use the timestamp of the latest candle
                price=current_close,
                rule_id=RULE_ID,
            ))

    return signals