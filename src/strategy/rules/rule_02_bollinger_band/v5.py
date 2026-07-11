from __future__ import annotations

import statistics
import numpy as np  # Available, but not strictly used for mean/stdev as statistics module is sufficient

from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle
from datetime import datetime

# --- Constants ---
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0
VOLUME_AVG_PERIOD = 50
VOLUME_MULTIPLIER = 1.5

# Minimum candles needed for Bollinger Bands (BB_PERIOD) and Volume Average (VOLUME_AVG_PERIOD)
# We need at least max(BB_PERIOD, VOLUME_AVG_PERIOD) candles to calculate the indicators
# for the *latest* warm candle.
MIN_CANDLES = max(BB_PERIOD, VOLUME_AVG_PERIOD)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on Bollinger Bands with Volume Spike confirmation.

    This rule detects potential trend reversals when the price moves significantly outside
    its typical range (Bollinger Bands) and is confirmed by a significant spike in trading volume.

    A Buy signal is generated when the latest warm candle's close price drops below the
    lower Bollinger Band AND its volume significantly increases (current volume > average volume * VOLUME_MULTIPLIER).

    A Sell signal is generated when the latest warm candle's close price rises above the
    upper Bollinger Band AND its volume significantly increases (current volume > average volume * VOLUME_MULTIPLIER).

    The hypothesis is that extreme price movements, when accompanied by high trading volume,
    provide stronger confirmation of potential reversals due to increased market participation.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for all calculations.
        # The 'hot' data (live ticks) is not directly used for the price/volume conditions
        # in this rule, as the pseudocode refers to 'close' prices of candles.
        if len(pair_data.warm) < MIN_CANDLES:
            continue

        # Extract close prices and volumes from warm candles
        closes = [c.close for c in pair_data.warm]
        volumes = [c.volume for c in pair_data.warm]

        # Get the latest warm candle's data for current conditions and signal timestamp
        latest_candle: WarmCandle = pair_data.warm[-1]
        current_close_price = latest_candle.close
        current_volume = latest_candle.volume
        signal_timestamp: datetime = latest_candle.hour

        # --- Calculate Bollinger Bands ---
        # Use the last BB_PERIOD closes for SMA and StdDev
        bb_closes = closes[-BB_PERIOD:]

        # Calculate the Simple Moving Average (SMA)
        bb_mean = statistics.mean(bb_closes)

        # Calculate the Standard Deviation (StdDev)
        # Handle cases where std dev might be zero (e.g., all prices are identical for BB_PERIOD)
        # statistics.stdev requires at least two data points. BB_PERIOD is 20, so this is fine.
        # If all values are identical, std dev is 0.
        if len(set(bb_closes)) < 2 and BB_PERIOD > 1:
            std = 0.0
        else:
            std = statistics.stdev(bb_closes)

        upper_band = bb_mean + BB_STD_DEV_MULTIPLIER * std
        lower_band = bb_mean - BB_STD_DEV_MULTIPLIER * std

        # --- Calculate Average Volume and Volume Spike Condition ---
        # Use the last VOLUME_AVG_PERIOD volumes for SMA
        avg_volume_values = volumes[-VOLUME_AVG_PERIOD:]

        # Calculate the Simple Moving Average (SMA) for volume
        avg_volume = statistics.mean(avg_volume_values)

        # Volume spike condition: current volume must be significantly higher than its average
        volume_spike_condition = (current_volume > avg_volume * VOLUME_MULTIPLIER)

        # --- Signal Generation ---
        # A buy signal if price drops below the lower band and volume spikes
        if current_close_price < lower_band and volume_spike_condition:
            signals.append(BuySignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=current_close_price,
                rule_id="b39f5bab-9223-4581-91a5-f15c80d1f25f"
            ))
        # A sell signal if price rises above the upper band and volume spikes
        elif current_close_price > upper_band and volume_spike_condition:
            signals.append(SellSignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=current_close_price,
                rule_id="b39f5bab-9223-4581-91a5-f15c80d1f25f"
            ))

    return signals