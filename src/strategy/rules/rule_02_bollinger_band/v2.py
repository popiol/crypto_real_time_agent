"""Bollinger Band V2: Volume-Filtered Entry.

This rule enhances rule_02_bollinger_band_v1 by adding a volume filter.
A buy signal is only emitted if the price breaches the lower Bollinger Band
AND the current trading volume is above its recent average.
Similarly, a sell signal is only emitted if the price breaches the upper Bollinger Band
AND the current trading volume is above its recent average.
This aims to confirm price breakouts from the bands with significant market participation,
reducing false signals in low-volume chop.
"""

from __future__ import annotations

import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# Constants for the rule
BB_WINDOW = 20  # Window for Bollinger Band calculations (prices)
K = 2.0         # Standard deviation multiplier for Bollinger Bands
VOL_WINDOW = 20 # Window for average volume calculations

# Minimum number of warm candles required for calculations
MIN_WARM_CANDLES = max(BB_WINDOW, VOL_WINDOW)

# Unique identifier for this rule
RULE_ID = "9544d721-68ef-4849-afc8-b922e46ce685"


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for both BB and volume calculations
        # and at least one hot tick for the current price and timestamp.
        if not pair_data.hot or len(pair_data.warm) < MIN_WARM_CANDLES:
            continue

        # Extract relevant data from warm candles.
        # pair_data.warm is ordered from oldest to newest.
        # We need the last BB_WINDOW closes for Bollinger Band calculation.
        # We need the last VOL_WINDOW volumes for average volume calculation.
        # Slicing with `[-N:]` correctly gets the last N elements.
        recent_closes = [c.close for c in pair_data.warm[-BB_WINDOW:]]
        recent_volumes = [c.volume for c in pair_data.warm[-VOL_WINDOW:]]

        # Calculate Bollinger Bands parameters (SMA and Standard Deviation)
        # statistics.stdev requires at least two data points. BB_WINDOW >= 20 ensures this.
        sma = statistics.mean(recent_closes)
        try:
            std = statistics.stdev(recent_closes)
        except statistics.StatisticsError:
            # This can happen if all values in recent_closes are identical and BB_WINDOW was 1,
            # or if the list is empty/single element. Given BB_WINDOW=20 and MIN_WARM_CANDLES check,
            # this specific error is unlikely for a non-zero length list.
            # If all prices are identical, stdev correctly returns 0.0.
            std = 0.0

        # Handle cases where standard deviation is zero (all prices are the same in the window)
        if std == 0:
            continue

        upper_band = sma + (std * K)
        lower_band = sma - (std * K)

        # Calculate average volume over the specified window
        avg_volume = statistics.mean(recent_volumes)

        # Get current price, current volume, and timestamp for the signal
        # Current price is from the latest tick
        current_price = pair_data.hot[-1].last_price
        # Current volume is from the latest *completed* warm candle
        current_volume = pair_data.warm[-1].volume
        ts = pair_data.hot[-1].polled_at

        # Check for buy signal with the volume filter
        if current_price < lower_band and current_volume > avg_volume:
            signals.append(BuySignal(
                pair=pair,
                timestamp=ts,
                price=current_price,
                rule_id=RULE_ID
            ))
        # Check for sell signal with the volume filter
        elif current_price > upper_band and current_volume > avg_volume:
            signals.append(SellSignal(
                pair=pair,
                timestamp=ts,
                price=current_price,
                rule_id=RULE_ID
            ))

    return signals