from __future__ import annotations
import numpy as np
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick

# --- PARAMETERS ---
PERIOD_BB = 20
"""Bollinger Band period for closing prices (number of warm candles)."""

STD_DEV_MULTIPLIER = 2.0
"""Standard deviation multiplier for Bollinger Bands."""

VOLUME_PERIOD = 20
"""Moving average period for volume confirmation (number of hot ticks)."""

VOLUME_MULTIPLIER = 1.5
"""Multiplier for average volume to confirm a signal."""

# Minimum data points required for calculations
MIN_CANDLES_FOR_BB = PERIOD_BB
MIN_TICKS_FOR_VOLUME_MA = VOLUME_PERIOD

# --- ASSUMPTION ON VOLUME DATA ---
# The provided WarmCandle model does not contain volume information.
# The pseudocode for this rule explicitly requires calculating a moving average of 'volumes'
# over `volume_period` for confirmation.
# To implement the rule as described with available data, we make the following assumption:
# 1. 'current_volume' refers to the `volume_24h` field of the most recent Tick in `pair_data.hot`.
# 2. 'AvgVolume' refers to the Simple Moving Average of the `volume_24h` field from the
#    last `VOLUME_PERIOD` Ticks in `pair_data.hot`.
# This means Bollinger Bands are calculated on `warm` candles, while volume confirmation
# uses `hot` tick data. This is a pragmatic choice given the data model limitations.
# If WarmCandle were to include a 'volume' field (e.g., hourly volume), it would be
# more consistent to use that for volume confirmation over the same `volume_period` as `PERIOD_BB`.
# ----------------------------------

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the "Enhance Bollinger Band with Volume Confirmation" trading rule.

    A Buy signal is generated when the price drops below the lower Bollinger Band
    AND the current trading volume (24h rolling) exceeds a defined moving average
    of recent 24h rolling volumes.
    A Sell signal is issued when the price rises above the upper Bollinger Band
    AND the current trading volume (24h rolling) is significantly above its average.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm candles for Bollinger Bands calculation
        if len(pair_data.warm) < MIN_CANDLES_FOR_BB:
            continue

        # Ensure enough hot ticks for volume moving average calculation
        if len(pair_data.hot) < MIN_TICKS_FOR_VOLUME_MA:
            continue

        # --- 1. Calculate Bollinger Bands ---
        # Use the last `PERIOD_BB` warm candles for BB calculation
        # np.array is more efficient for numerical operations
        closes = np.array([c.close for c in pair_data.warm[-PERIOD_BB:]])

        # Calculate mean and standard deviation of closing prices
        mean_bb = np.mean(closes)
        std_bb = np.std(closes) # Default is population standard deviation

        if std_bb == 0:
            # If standard deviation is zero, bands collapse to the mean,
            # indicating no price movement in the period, making signals meaningless.
            continue

        upper_bb = mean_bb + STD_DEV_MULTIPLIER * std_bb
        lower_bb = mean_bb - STD_DEV_MULTIPLIER * std_bb

        # --- 2. Calculate Volume Confirmation ---
        # Use the last `VOLUME_PERIOD` hot ticks for volume moving average
        volumes_24h = np.array([t.volume_24h for t in pair_data.hot[-VOLUME_PERIOD:]])
        
        # Calculate Simple Moving Average of the 24-hour rolling volume
        avg_volume_24h = np.mean(volumes_24h)

        # Get current price and 24-hour rolling volume from the most recent tick
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_volume_24h = current_tick.volume_24h
        timestamp = current_tick.polled_at

        # Calculate the threshold for volume confirmation
        volume_threshold = avg_volume_24h * VOLUME_MULTIPLIER

        # --- 3. Generate Signals ---
        # BUY SIGNAL: current_price < LowerBB AND current_volume > (AvgVolume * volume_multiplier)
        if current_price < lower_bb and current_volume_24h > volume_threshold:
            signals.append(BuySignal(pair=pair, timestamp=timestamp, price=current_price))
        # SELL SIGNAL: current_price > UpperBB AND current_volume > (AvgVolume * volume_multiplier)
        elif current_price > upper_bb and current_volume_24h > volume_threshold:
            signals.append(SellSignal(pair=pair, timestamp=timestamp, price=current_price))

    return signals