"""Rule 03 — Asymmetric Bollinger Bands with Volume Confirmation (v4)."""
from __future__ import annotations

import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# Default asymmetric K multipliers
K_BUY = 1.8  # Multiplier for the lower band (buy signal sensitivity)
K_SELL = 2.2 # Multiplier for the upper band (sell signal sensitivity)

# Minimum number of warm candles required for Bollinger Band calculation
MIN_BB_CANDLES = 10

# Lookback window for calculating average volume from warm candles
VOLUME_AVG_WINDOW = 5

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    # Determine the minimum number of warm candles needed for both BB and volume calculations
    REQUIRED_WARM_CANDLES = max(MIN_BB_CANDLES, VOLUME_AVG_WINDOW)

    for pair, pair_data in data.items():
        # Ensure we have enough historical warm candles and at least one hot tick for current price/timestamp
        if len(pair_data.warm) < REQUIRED_WARM_CANDLES or not pair_data.hot:
            continue

        # Extract closing prices and volumes from warm candles
        # These represent historical hourly data
        closes = [c.close for c in pair_data.warm]
        volumes = [c.volume for c in pair_data.warm]

        # Calculate Moving Average (MA) for prices
        mean_price = statistics.mean(closes)

        # Calculate Standard Deviation (STD) for prices
        # statistics.stdev requires at least two data points. MIN_BB_CANDLES ensures this.
        std_price = statistics.stdev(closes)

        # If standard deviation is zero, prices haven't moved, so no band deviation is possible.
        if std_price == 0:
            continue

        # Get current price and timestamp from the most recent tick
        current_tick: Tick = pair_data.hot[-1]
        current_price: float = current_tick.last_price
        ts: datetime = current_tick.polled_at

        # Get current volume from the last completed warm candle
        # This aligns with using historical warm candle data for volume averages
        current_volume: float = volumes[-1]

        # Calculate asymmetric Bollinger Bands
        lower_band_asymmetric = mean_price - K_BUY * std_price
        upper_band_asymmetric = mean_price + K_SELL * std_price

        # Calculate average volume over the specified window from warm candles
        # Ensure we have enough data for the volume window
        recent_volumes = volumes[-VOLUME_AVG_WINDOW:]
        avg_volume = statistics.mean(recent_volumes)

        # Generate signals based on current price relative to asymmetric bands
        # AND confirmed by current volume being above the recent average volume
        if current_price < lower_band_asymmetric and current_volume > avg_volume:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif current_price > upper_band_asymmetric and current_volume > avg_volume:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals