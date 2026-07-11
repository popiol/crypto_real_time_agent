from __future__ import annotations

import numpy as np
from datetime import datetime # Required for BuySignal/SellSignal timestamp type

from src.agent.models import BuySignal, MarketData, SellSignal

# --- Rule Parameters ---
# The period for calculating the Simple Moving Average (SMA) and Standard Deviation (STDDEV).
# This is kept the same as the underlying rule's implied period but explicitly set to 20
# as per the pseudocode for the modification.
PERIOD = 20

# The standard deviation multiplier for the Bollinger Bands.
# This is the key adjustment: increased from 2.0 (original) to 2.2 for wider bands.
STD_DEV_MULTIPLIER = 2.2


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Bollinger Band V1 Sensitivity Adjustment trading rule.

    This rule generates buy signals when the current price drops below the lower
    Bollinger Band, indicating a potential oversold condition and a mean-reversion
    opportunity upwards. It generates sell signals when the current price rises
    above the upper Bollinger Band, indicating a potential overbought condition
    and a mean-reversion opportunity downwards.

    This version adjusts the standard deviation multiplier to 2.2, making the
    bands wider compared to the original rule (which used 2.0). This aims to
    reduce sensitivity and filter out minor price fluctuations, focusing on
    more significant mean-reversion opportunities.

    Args:
        data: A MarketData object containing tick and candle data for various
              currency pairs.

    Returns:
        A list of BuySignal or SellSignal objects indicating trading opportunities.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candle data to calculate Bollinger Bands for the specified period.
        # The pseudocode specifies 'period = 20'.
        if len(pair_data.warm) < PERIOD:
            continue

        # Extract closing prices for the last 'PERIOD' candles.
        # This ensures the calculation window is consistent with the `PERIOD` parameter.
        closes = np.array([c.close for c in pair_data.warm[-PERIOD:]])

        # Calculate the Middle Band (MB) as the Simple Moving Average (SMA) of closes.
        middle_band = np.mean(closes)

        # Calculate the Standard Deviation (STDDEV) of closes.
        # np.std calculates the population standard deviation by default, which is
        # standard practice for Bollinger Bands.
        std_dev = np.std(closes)

        # If standard deviation is zero, all prices in the window are identical.
        # This makes the bands collapse, rendering signals meaningless. Skip such cases.
        if std_dev == 0:
            continue

        # Calculate the Upper Band (UB) and Lower Band (LB) using the adjusted multiplier.
        upper_band = middle_band + (STD_DEV_MULTIPLIER * std_dev)
        lower_band = middle_band - (STD_DEV_MULTIPLIER * std_dev)

        # Ensure there is hot (tick) data to get the current price for comparison.
        if not pair_data.hot:
            continue

        # Get the current price and timestamp from the most recent tick.
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        timestamp = current_tick.polled_at

        # Generate buy or sell signals based on price crossing the bands.
        # Buy Signal: Current price falls below the lower band (oversold).
        if current_price < lower_band:
            signals.append(BuySignal(pair=pair, timestamp=timestamp, price=current_price))
        # Sell Signal: Current price rises above the upper band (overbought).
        elif current_price > upper_band:
            signals.append(SellSignal(pair=pair, timestamp=timestamp, price=current_price))

    return signals