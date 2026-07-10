"""Rule 09 — Bollinger Band Reversion with Inside Close Confirmation."""
from __future__ import annotations

import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Parameters for Bollinger Bands
PERIOD = 20  # Lookback period for Bollinger Bands
STD_DEV_MULTIPLIER = 2.0  # Standard deviation multiplier

# Minimum candles needed for calculation:
# We need `PERIOD` candles to calculate the Bollinger Bands.
# These bands are applied to the `previous_close`.
# Then, we need one additional candle (`current_candle`) to check for the "inside close" confirmation.
# So, we need at least `PERIOD + 1` warm candles in total.
MIN_CANDLES = PERIOD + 1


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candle data for calculation and confirmation
        if not pair_data.warm or len(pair_data.warm) < MIN_CANDLES:
            continue

        # Get the relevant candles from the end of the warm data list
        # current_candle is the most recent completed candle
        current_candle: WarmCandle = pair_data.warm[-1]
        # previous_candle is the candle immediately preceding the current_candle
        previous_candle: WarmCandle = pair_data.warm[-2]

        # Bollinger Bands are calculated based on closing prices *up to* the previous_candle.close.
        # This means we take the `PERIOD` number of closes ending with `previous_candle.close`.
        # The slice `pair_data.warm[-(PERIOD + 1):-1]` correctly selects these `PERIOD` candles.
        closes_for_bb = [c.close for c in pair_data.warm[-(PERIOD + 1):-1]]

        # Calculate mean for Bollinger Bands
        mean = statistics.mean(closes_for_bb)

        # Calculate standard deviation. Handle cases with insufficient data or zero std dev.
        # statistics.stdev requires at least 2 data points.
        if len(closes_for_bb) < 2:
            std = 0.0
        else:
            std = statistics.stdev(closes_for_bb)

        # If standard deviation is zero, the bands collapse to the mean, making signals invalid.
        if std == 0:
            continue

        # Calculate Bollinger Bands (Upper and Lower)
        upper_band = mean + STD_DEV_MULTIPLIER * std
        lower_band = mean - STD_DEV_MULTIPLIER * std

        # Extract closing prices for the current and previous candles
        current_close = current_candle.close
        previous_close = previous_candle.close

        # The timestamp for the signal is the 'hour' of the current (confirming) candle.
        signal_timestamp: datetime = current_candle.hour

        # Buy Signal Logic:
        # 1. Price must have dipped below the lower Bollinger Band (previous_close < LowerBand).
        # 2. The current candle must close back *above* the lower Bollinger Band (current_close > LowerBand).
        if previous_close < lower_band and current_close > lower_band:
            signals.append(BuySignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=current_close,
                rule_id="009d3a5f-6695-4bfb-9643-05bbaafc3cae"
            ))

        # Sell Signal Logic:
        # 1. Price must have peaked above the upper Bollinger Band (previous_close > UpperBand).
        # 2. The current candle must close back *below* the upper Bollinger Band (current_close < UpperBand).
        elif previous_close > upper_band and current_close < upper_band:
            signals.append(SellSignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=current_close,
                rule_id="009d3a5f-6695-4bfb-9643-05bbaafc3cae"
            ))

    return signals