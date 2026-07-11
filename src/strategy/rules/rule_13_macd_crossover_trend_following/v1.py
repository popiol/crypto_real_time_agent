from __future__ import annotations
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal

# Rule-specific constants
FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9

# Minimum number of candles required to calculate MACD and Signal line.
# This ensures that the slowest EMA (SLOW_PERIOD) has had enough data to
# become somewhat stable, and subsequently, the Signal Line (EMA of MACD)
# also has sufficient data points to be meaningful for a crossover check.
# A conservative estimate is SLOW_PERIOD + SIGNAL_PERIOD.
MIN_CANDLES = SLOW_PERIOD + SIGNAL_PERIOD 

def _calculate_ema_series(prices: list[float], period: int) -> list[float]:
    """
    Calculates Exponential Moving Average (EMA) for a series of prices.
    The first EMA value is initialized with the first price in the series.
    """
    if not prices:
        return []

    ema_series = []
    multiplier = 2 / (period + 1)
    
    # Initialize the first EMA with the first price
    ema_series.append(prices[0]) 

    # Calculate subsequent EMAs
    for i in range(1, len(prices)):
        ema = (prices[i] * multiplier) + (ema_series[-1] * (1 - multiplier))
        ema_series.append(ema)
    
    return ema_series

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates Buy/Sell signals based on MACD line crossing above/below its signal line.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough warm candles are available for MACD calculation
        if len(warm_candles) < MIN_CANDLES:
            continue

        # Extract close prices from the warm candles
        close_prices = [candle.close for candle in warm_candles]

        # Calculate the Fast and Slow EMAs of the close prices
        ema_fast_series = _calculate_ema_series(close_prices, FAST_PERIOD)
        ema_slow_series = _calculate_ema_series(close_prices, SLOW_PERIOD)

        # Calculate the MACD Line (Fast EMA - Slow EMA)
        macd_line_series = [
            ema_fast_series[i] - ema_slow_series[i]
            for i in range(len(close_prices))
        ]

        # Calculate the Signal Line (EMA of the MACD Line)
        signal_line_series = _calculate_ema_series(macd_line_series, SIGNAL_PERIOD)
        
        # Ensure we have at least two points for both MACD and Signal lines
        # to detect a crossover. This check is primarily for robustness,
        # as MIN_CANDLES should already ensure this.
        if len(macd_line_series) < 2 or len(signal_line_series) < 2:
            continue

        # Get the current and previous values for MACD and Signal lines
        macd_current = macd_line_series[-1]
        macd_prev = macd_line_series[-2]

        signal_current = signal_line_series[-1]
        signal_prev = signal_line_series[-2]

        # Get the timestamp and price of the latest candle for signal generation
        latest_candle = warm_candles[-1]
        signal_timestamp = latest_candle.hour
        signal_price = latest_candle.close

        # Check for crossover signals
        # Buy signal: MACD line crosses above the Signal line
        if macd_prev < signal_prev and macd_current > signal_current:
            signals.append(BuySignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
            ))
        # Sell signal: MACD line crosses below the Signal line
        elif macd_prev > signal_prev and macd_current < signal_current:
            signals.append(SellSignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
            ))

    return signals