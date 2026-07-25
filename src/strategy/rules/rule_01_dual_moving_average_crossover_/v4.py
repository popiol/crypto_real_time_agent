from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal

# Constants for EMA periods
SHORT_EMA_PERIOD = 9
LONG_EMA_PERIOD = 18

def _calculate_ema(prices: list[float], period: int) -> list[float]:
    """
    Calculates Exponential Moving Average (EMA) for a given list of prices.
    The first EMA value is typically initialized as the Simple Moving Average (SMA)
    of the first 'period' prices. Subsequent EMAs are calculated using the standard formula.

    Args:
        prices: A list of float values representing chronological prices (e.g., closing prices).
        period: The number of periods to use for the EMA calculation.

    Returns:
        A list of float values representing the EMA series. Returns an empty list
        if there isn't enough data to calculate at least one EMA value.
    """
    if len(prices) < period:
        return []

    ema_values = []
    
    # Calculate the initial EMA value as an SMA of the first 'period' prices.
    # This is a common practice to start the EMA series.
    initial_ema = np.mean(prices[:period])
    ema_values.append(initial_ema)

    # Calculate the smoothing factor (K)
    k = 2 / (period + 1)

    # Calculate subsequent EMA values using the formula:
    # EMA_current = (Price_current - EMA_previous) * K + EMA_previous
    for i in range(period, len(prices)):
        current_price = prices[i]
        previous_ema = ema_values[-1]
        current_ema = (current_price - previous_ema) * k + previous_ema
        ema_values.append(current_ema)
        
    return ema_values

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Responsive EMA Crossover trading rule.

    A buy signal is generated when the 9-period EMA crosses above the 18-period EMA.
    A sell signal is generated when the 9-period EMA crosses below the 18-period EMA.
    Calculations are based on `data.warm` (hourly candles).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_data = pair_data.warm

        # We need at least (LONG_EMA_PERIOD + 1) candles to calculate two EMA values
        # for the long EMA. Two values are necessary to detect a crossover (previous and current).
        required_data_points = LONG_EMA_PERIOD + 1
        if len(warm_data) < required_data_points:
            # Not enough hourly data to calculate the required EMA series for crossover detection.
            continue

        # Extract 'close' prices from each WarmCandle object.
        # These are assumed to be chronologically ordered, with the latest price at the end.
        prices = [candle.close for candle in warm_data]

        # Calculate EMA series for both short and long periods
        short_ema_series = _calculate_ema(prices, SHORT_EMA_PERIOD)
        long_ema_series = _calculate_ema(prices, LONG_EMA_PERIOD)

        # Ensure both EMA series have at least 2 values to detect a crossover.
        # If _calculate_ema returned an empty list due to insufficient data for its period,
        # this check will catch it.
        if len(short_ema_series) < 2 or len(long_ema_series) < 2:
            continue

        # Get current and previous EMA values from the end of each series
        current_short_ema = short_ema_series[-1]
        previous_short_ema = short_ema_series[-2]
        current_long_ema = long_ema_series[-1]
        previous_long_ema = long_ema_series[-2]

        # Determine the most recent price and timestamp for the signal.
        # Prioritize 'hot' (tick) data for the most up-to-date information,
        # otherwise fall back to the latest 'warm' (hourly candle) data.
        current_price = None
        timestamp = None

        if pair_data.hot:
            current_price = pair_data.hot[-1].last_price
            timestamp = pair_data.hot[-1].polled_at
        elif pair_data.warm:
            current_price = pair_data.warm[-1].close
            timestamp = pair_data.warm[-1].hour
        else:
            # No current price available from hot or warm data. Cannot generate a relevant signal.
            continue
            
        # Determine trading signal based on the EMA crossover logic
        # Buy Signal: Short EMA crosses above Long EMA
        # This occurs when the current short EMA is above the current long EMA,
        # AND the previous short EMA was at or below the previous long EMA.
        if current_short_ema > current_long_ema and previous_short_ema <= previous_long_ema:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id="EMA_Crossover_Responsive_v1"
            ))
        # Sell Signal: Short EMA crosses below Long EMA
        # This occurs when the current short EMA is below the current long EMA,
        # AND the previous short EMA was at or above the previous long EMA.
        elif current_short_ema < current_long_ema and previous_short_ema >= previous_long_ema:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id="EMA_Crossover_Responsive_v1"
            ))

    return signals