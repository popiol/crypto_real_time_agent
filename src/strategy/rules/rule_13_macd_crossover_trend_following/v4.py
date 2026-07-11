from __future__ import annotations
from datetime import datetime
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Rule-specific constants
FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9
VOLUME_AVG_PERIOD = 20  # Parameter for volume moving average
# Key modification: Relaxed volume confirmation factor from 0.8 to 0.5
VOLUME_CONFIRMATION_FACTOR = 0.5

# Minimum number of candles required.
# This ensures that the slowest EMA (SLOW_PERIOD) has had enough data to
# become somewhat stable, and subsequently, the Signal Line (EMA of MACD)
# also has sufficient data points to be meaningful for a crossover check.
# Also ensures enough data for the Volume Moving Average.
MIN_CANDLES = max(SLOW_PERIOD + SIGNAL_PERIOD, VOLUME_AVG_PERIOD)

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

def _calculate_sma_series(data: list[float], period: int) -> list[float]:
    """
    Calculates Simple Moving Average (SMA) for a series of data using numpy.
    Returns a series of the same length as the input, padding initial values
    with the first calculated SMA to align indices.
    """
    if not data:
        return []
    if period <= 0:
        raise ValueError("Period must be greater than 0")
    
    if len(data) < period:
        # If not enough data for a full SMA window, return a list of zeros
        # to ensure the list has the correct length. The MIN_CANDLES check
        # in the main signal function should prevent this state from being
        # reached if there isn't enough data overall.
        return [0.0] * len(data) 

    np_data = np.array(data)
    weights = np.ones(period) / period
    
    # Use 'valid' mode to get only the fully calculated SMAs
    sma_valid = np.convolve(np_data, weights, 'valid')
    
    # Pad the beginning of the SMA series to match the length of the input data.
    # The 'edge' mode pads with the values at the edge of the array.
    # So, sma_valid[0] will be repeated 'period - 1' times at the beginning.
    padded_sma = np.pad(sma_valid, (period - 1, 0), mode='edge')
    
    return padded_sma.tolist()


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates Buy/Sell signals based on MACD line crossing above/below its signal line,
    confirmed by relaxed elevated trading volume (at least 50% of its moving average).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough warm candles are available for all calculations
        if len(warm_candles) < MIN_CANDLES:
            continue

        # Extract close prices and volumes from the warm candles
        close_prices = [candle.close for candle in warm_candles]
        volumes = [candle.volume for candle in warm_candles] # Extract volumes

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
        
        # Calculate Volume Moving Average
        volume_ma_series = _calculate_sma_series(volumes, VOLUME_AVG_PERIOD)

        # Ensure we have at least two points for MACD, Signal lines, and at least one for Volume MA.
        # MIN_CANDLES should primarily ensure this, but these checks add robustness.
        if len(macd_line_series) < 2 or len(signal_line_series) < 2 or len(volume_ma_series) < 1:
            continue

        # Get the current and previous values for MACD and Signal lines
        macd_current = macd_line_series[-1]
        macd_prev = macd_line_series[-2]

        signal_current = signal_line_series[-1]
        signal_prev = signal_line_series[-2]
        
        # Get current volume and its moving average
        current_volume = volumes[-1]
        volume_ma_current = volume_ma_series[-1]

        # Get the timestamp and price of the latest candle for signal generation
        latest_candle = warm_candles[-1]
        signal_timestamp = latest_candle.hour
        signal_price = latest_candle.close

        # Check for crossover signals with relaxed volume confirmation
        # Buy signal: MACD line crosses above the Signal line AND current volume > (Volume MA * VOLUME_CONFIRMATION_FACTOR)
        if (macd_prev < signal_prev and macd_current > signal_current and
            current_volume > (volume_ma_current * VOLUME_CONFIRMATION_FACTOR)):
            signals.append(BuySignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
            ))
        # Sell signal: MACD line crosses below the Signal line AND current volume > (Volume MA * VOLUME_CONFIRMATION_FACTOR)
        elif (macd_prev > signal_prev and macd_current < signal_current and
              current_volume > (volume_ma_current * VOLUME_CONFIRMATION_FACTOR)):
            signals.append(SellSignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
            ))

    return signals