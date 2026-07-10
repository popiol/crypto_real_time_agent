from __future__ import annotations

import statistics

from src.agent.models import BuySignal, MarketData, SellSignal

# Constants for the Bollinger Band calculation
PERIOD = 20  # Default period for SMA and STD, as per pseudocode
STD_DEV_MULTIPLIER = 2.5  # Standard deviation multiplier for wider bands, as per idea description

# Minimum number of warm candles required to perform the calculation.
# Must be at least `PERIOD` to have enough data for SMA/STD.
MIN_CANDLES_REQUIRED = PERIOD


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements a Bollinger Band Mean Reversion strategy with wider bands (2.5 SD).

    A Buy signal is emitted when the price drops below the lower 2.5-SD Bollinger Band.
    A Sell signal is emitted when the price rises above the upper 2.5-SD Bollinger Band.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for the Bollinger Band calculation
        if len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue

        # Ensure we have recent tick data to determine current price and timestamp
        if not pair_data.hot:
            continue

        # Extract the closing prices for the last `PERIOD` candles
        # We slice [-PERIOD:] to get the most recent `PERIOD` candles.
        closes = [c.close for c in pair_data.warm[-PERIOD:]]

        # Calculate the Simple Moving Average (SMA)
        mean = statistics.mean(closes)

        # Calculate the Standard Deviation (STD)
        # statistics.stdev requires at least two data points.
        # Since MIN_CANDLES_REQUIRED is PERIOD (20), len(closes) will be 20,
        # so statistics.stdev will not raise an error due to insufficient data points.
        # If all closing prices are identical, std will be 0. In this case, the bands
        # collapse to the mean, and signals become trivial or misleading, so we skip.
        std = statistics.stdev(closes)
        if std == 0:
            continue

        # Calculate the upper and lower Bollinger Bands
        upper_band = mean + (std * STD_DEV_MULTIPLIER)
        lower_band = mean - (std * STD_DEV_MULTIPLIER)

        # Get the current price and timestamp from the most recent tick
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        timestamp = current_tick.polled_at

        # Generate signals based on current price relative to the Bollinger Bands
        if current_price < lower_band:
            signals.append(BuySignal(pair=pair, timestamp=timestamp, price=current_price))
        elif current_price > upper_band:
            signals.append(SellSignal(pair=pair, timestamp=timestamp, price=current_price))

    return signals