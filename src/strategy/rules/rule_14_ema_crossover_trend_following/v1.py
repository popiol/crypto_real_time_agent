from __future__ import annotations
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Constants for EMA periods
SHORT_EMA_PERIOD = 12
LONG_EMA_PERIOD = 21  # Adjusted from 26 to be compatible with max 24 warm candles.
                      # With 24 candles, a 21-period EMA allows for at least 4 calculated values.

# To detect a crossover, we need at least two consecutive calculated EMA values for both short and long periods.
# This means we need (period + 1) candles to obtain two EMA values for that period.
# Therefore, the minimum number of candles required is max(SHORT_EMA_PERIOD + 1, LONG_EMA_PERIOD + 1).
MIN_CANDLES_REQUIRED = max(SHORT_EMA_PERIOD + 1, LONG_EMA_PERIOD + 1)

RULE_ID = "EMA_Crossover_Trend_Following_v1"

def _calculate_ema(prices: list[float], period: int) -> list[float]:
    """
    Calculates Exponential Moving Average for a list of prices.
    Returns a list containing only the valid, calculated EMA values.
    """
    if len(prices) < period:
        return []

    ema_values_calculated = []
    
    # Initialize the first EMA with the Simple Moving Average (SMA) of the first 'period' prices
    current_ema = sum(prices[:period]) / period
    ema_values_calculated.append(current_ema)
    
    multiplier = 2 / (period + 1)
    
    # Calculate subsequent EMA values
    for i in range(period, len(prices)):
        current_ema = (prices[i] - current_ema) * multiplier + current_ema
        ema_values_calculated.append(current_ema)
        
    return ema_values_calculated

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates buy and sell signals based on the crossover of two Exponential Moving Averages (EMAs).

    A buy signal is emitted when the shorter-term EMA crosses above the longer-term EMA,
    indicating an upward trend.
    A sell signal is emitted when the shorter-term EMA crosses below the longer-term EMA,
    signaling a downward trend.

    Args:
        data (MarketData): A dictionary containing market data for various currency pairs.

    Returns:
        list[BuySignal | SellSignal]: A list of generated buy or sell signals.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough warm candles are available for EMA calculation and crossover detection.
        # We need at least MIN_CANDLES_REQUIRED to get two valid EMA values for both periods.
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract closing prices from the warm candles
        close_prices = [candle.close for candle in warm_candles]

        # Calculate EMA values for both short and long periods
        ema_short_values = _calculate_ema(close_prices, SHORT_EMA_PERIOD)
        ema_long_values = _calculate_ema(close_prices, LONG_EMA_PERIOD)

        # We need at least two calculated EMA values for both to detect a crossover (current and previous states).
        if len(ema_short_values) < 2 or len(ema_long_values) < 2:
            continue
            
        # Get the latest (current) and previous EMA values for both periods
        ema_short_current = ema_short_values[-1]
        ema_short_prev = ema_short_values[-2]

        ema_long_current = ema_long_values[-1]
        ema_long_prev = ema_long_values[-2]

        # Get the latest candle for its timestamp and closing price, which will be used for the signal
        latest_candle = warm_candles[-1]
        
        # Check for crossover conditions:
        # Buy signal: Short EMA crosses above Long EMA.
        # This occurs if the previous short EMA was below the long EMA, and the current short EMA is above the long EMA.
        if ema_short_prev < ema_long_prev and ema_short_current > ema_long_current:
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_candle.hour,
                price=latest_candle.close,
                rule_id=RULE_ID
            ))
        # Sell signal: Short EMA crosses below Long EMA.
        # This occurs if the previous short EMA was above the long EMA, and the current short EMA is below the long EMA.
        elif ema_short_prev > ema_long_prev and ema_short_current < ema_long_current:
            signals.append(SellSignal(
                pair=pair,
                timestamp=latest_candle.hour,
                price=latest_candle.close,
                rule_id=RULE_ID
            ))
            
    return signals