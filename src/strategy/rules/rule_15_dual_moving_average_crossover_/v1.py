from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle
from datetime import datetime

# Rule constants
SHORT_EMA_PERIOD = 12
LONG_EMA_PERIOD = 26
RULE_ID = "3345d733-a608-4374-b9c3-cd5add27d0fd"

# Minimum number of candles required to calculate current and previous EMAs for both periods.
# To obtain both the current and the previous EMA values, we need at least
# `max(SHORT_EMA_PERIOD, LONG_EMA_PERIOD)` data points to allow the EMA to "warm up"
# (though our _calculate_ema provides values from the first point)
# and then one additional point to get the 'previous' value.
# So, we need `max(SHORT_EMA_PERIOD, LONG_EMA_PERIOD) + 1` candles.
MIN_CANDLES_REQUIRED = max(SHORT_EMA_PERIOD, LONG_EMA_PERIOD) + 1


def _calculate_ema(prices: list[float], period: int) -> np.ndarray:
    """
    Calculates the Exponential Moving Average (EMA) for a given series of prices.
    The EMA is initialized with the first price in the series.
    """
    if not prices:
        return np.array([])

    prices_arr = np.array(prices, dtype=float)
    if period <= 0:
        raise ValueError("EMA period must be positive.")

    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices_arr, dtype=float)
    ema[0] = prices_arr[0]  # Initialize with the first price

    for i in range(1, len(prices_arr)):
        ema[i] = prices_arr[i] * alpha + ema[i - 1] * (1 - alpha)
    return ema


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Dual Moving Average Crossover with Price Confirmation' trading rule.

    A Buy signal is generated when the short EMA crosses above the long EMA,
    and the current price is above both EMAs.
    A Sell signal is generated when the short EMA crosses below the long EMA,
    and the current price is below both EMAs.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure sufficient historical data is available for EMA calculations
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract closing prices from warm candles
        close_prices = [candle.close for candle in warm_candles]

        # Calculate EMAs for both short and long periods
        ema_short = _calculate_ema(close_prices, SHORT_EMA_PERIOD)
        ema_long = _calculate_ema(close_prices, LONG_EMA_PERIOD)

        # Retrieve current and previous values for price and EMAs
        current_price = warm_candles[-1].close
        current_timestamp = warm_candles[-1].hour

        current_ema_short = ema_short[-1]
        current_ema_long = ema_long[-1]

        previous_ema_short = ema_short[-2]
        previous_ema_long = ema_long[-2]

        # Buy signal condition:
        # 1. Short EMA crosses above Long EMA
        # 2. Current price is above both EMAs (confirmation)
        if (previous_ema_short <= previous_ema_long and
            current_ema_short > current_ema_long and
            current_price > current_ema_short and
            current_price > current_ema_long):
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=RULE_ID,
            ))

        # Sell signal condition:
        # 1. Short EMA crosses below Long EMA
        # 2. Current price is below both EMAs (confirmation)
        elif (previous_ema_short >= previous_ema_long and
              current_ema_short < current_ema_long and
              current_price < current_ema_short and
              current_price < current_ema_long):
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=RULE_ID,
            ))

    return signals