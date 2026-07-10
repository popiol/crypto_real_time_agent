from __future__ import annotations
import statistics

from src.agent.models import BuySignal, MarketData, PairData, SellSignal

# Parameters as per pseudocode and common practice for Bollinger Bands
BB_PERIOD = 20  # Period for Bollinger Band SMA and StdDev (number of warm candles)
BB_STD_DEV_MULTIPLIER = 2.0  # Standard deviation multiplier for Bollinger Bands
VOLUME_AVG_PERIOD = 20  # Period for calculating average volume (number of hot ticks)
VOLUME_MULTIPLIER = 1.5  # Multiplier for average volume threshold

# Minimum data required for calculations
MIN_CANDLES_FOR_BB = BB_PERIOD
MIN_TICKS_FOR_VOLUME = VOLUME_AVG_PERIOD


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm candles for Bollinger Bands calculation
        if len(pair_data.warm) < MIN_CANDLES_FOR_BB:
            continue

        # Ensure enough hot ticks for current price and volume average calculation
        if len(pair_data.hot) < MIN_TICKS_FOR_VOLUME:
            continue

        # --- Calculate Bollinger Bands ---
        # Get the 'close' prices for the most recent BB_PERIOD warm candles.
        # The `warm` list is ordered from oldest to newest.
        closes = [c.close for c in pair_data.warm[-BB_PERIOD:]]

        # Calculate Simple Moving Average (SMA) of close prices
        mean_close = statistics.mean(closes)
        # Calculate Standard Deviation (StdDev) of close prices
        std_dev_close = statistics.stdev(closes)

        # If standard deviation is zero, prices haven't moved, bands would be flat.
        # This makes the bands degenerate, so no meaningful signal.
        if std_dev_close == 0:
            continue

        # Calculate Upper and Lower Bollinger Bands
        upper_band = mean_close + (std_dev_close * BB_STD_DEV_MULTIPLIER)
        lower_band = mean_close - (std_dev_close * BB_STD_DEV_MULTIPLIER)

        # --- Calculate Volume Threshold ---
        # Get the 'volume_24h' from the most recent VOLUME_AVG_PERIOD hot ticks.
        # The `hot` list is ordered from oldest to newest.
        volumes_24h = [t.volume_24h for t in pair_data.hot[-VOLUME_AVG_PERIOD:]]

        # Calculate average 24-hour volume
        avg_volume_24h = statistics.mean(volumes_24h)
        # Calculate the volume threshold for confirmation
        volume_threshold = avg_volume_24h * VOLUME_MULTIPLIER

        # --- Get Current Data ---
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        # Using the latest 24h volume for confirmation
        current_volume = current_tick.volume_24h
        timestamp = current_tick.polled_at

        # --- Generate Signals ---
        # Buy signal: current price drops below lower band AND current volume is significantly above average
        if current_price < lower_band and current_volume > volume_threshold:
            signals.append(BuySignal(pair=pair, timestamp=timestamp, price=current_price))
        # Sell signal: current price rises above upper band AND current volume is significantly above average
        elif current_price > upper_band and current_volume > volume_threshold:
            signals.append(SellSignal(pair=pair, timestamp=timestamp, price=current_price))

    return signals