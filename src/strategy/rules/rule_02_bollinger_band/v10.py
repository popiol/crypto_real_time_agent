from __future__ import annotations

import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle


# Constants for Bollinger Bands calculation
BB_PERIOD = 20  # Number of warm candles for SMA and STDDEV
STD_DEV_MULTIPLIER = 2.0  # Multiplier for standard deviation to define bands

# Constants for Volume Confirmation
VOLUME_PERIOD = 20  # Number of hot ticks for average volume calculation
VOLUME_THRESHOLD_MULTIPLIER = 1.5  # Current volume must be X times above average volume

# Minimum data points required for calculations
MIN_CANDLES_FOR_BB = BB_PERIOD
MIN_TICKS_FOR_VOLUME = VOLUME_PERIOD


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm candle data for Bollinger Bands calculation
        if len(pair_data.warm) < MIN_CANDLES_FOR_BB:
            continue

        # Ensure enough hot tick data for volume average and current price/volume
        if len(pair_data.hot) < MIN_TICKS_FOR_VOLUME:
            continue

        # Extract closing prices for Bollinger Bands calculation
        # Use the last BB_PERIOD warm candles for SMA and STDDEV
        bb_closes = [c.close for c in pair_data.warm[-BB_PERIOD:]]

        # Calculate Simple Moving Average (SMA) of prices
        sma_price = statistics.mean(bb_closes)

        # Calculate Standard Deviation of prices
        # statistics.stdev requires at least 2 data points. If all prices are identical, std_price will be 0.
        # The MIN_CANDLES_FOR_BB check ensures len(bb_closes) is at least BB_PERIOD, so > 1.
        std_price = statistics.stdev(bb_closes)

        # If standard deviation is zero, it means no price movement, so no meaningful bands.
        # This prevents potential false signals or division by zero in more complex scenarios.
        if std_price == 0:
            continue

        # Calculate Upper and Lower Bollinger Bands
        upper_band = sma_price + (std_price * STD_DEV_MULTIPLIER)
        lower_band = sma_price - (std_price * STD_DEV_MULTIPLIER)

        # Extract 24-hour rolling volume from hot (tick) data for volume confirmation
        # Use the last VOLUME_PERIOD hot ticks for average volume
        volume_24h_data = [t.volume_24h for t in pair_data.hot[-VOLUME_PERIOD:]]

        # Calculate average volume
        avg_volume = statistics.mean(volume_24h_data)

        # Get current price and current 24-hour volume from the latest tick
        latest_tick = pair_data.hot[-1]
        current_price = latest_tick.last_price
        current_volume = latest_tick.volume_24h
        timestamp = latest_tick.polled_at

        # Check for Buy Signal:
        # Price drops below the lower Bollinger Band AND current volume is significantly above its recent average.
        if current_price < lower_band and current_volume > (avg_volume * VOLUME_THRESHOLD_MULTIPLIER):
            signals.append(BuySignal(pair=pair, timestamp=timestamp, price=current_price))
        # Check for Sell Signal:
        # Price rises above the upper Bollinger Band AND current volume is significantly above its recent average.
        elif current_price > upper_band and current_volume > (avg_volume * VOLUME_THRESHOLD_MULTIPLIER):
            signals.append(SellSignal(pair=pair, timestamp=timestamp, price=current_price))

    return signals