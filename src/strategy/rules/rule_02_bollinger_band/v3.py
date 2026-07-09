from __future__ import annotations
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# Parameters
PERIOD = 20  # e.g., 20 periods for SMA
STD_DEV_MULTIPLIER = 1.0  # Reduced from typical 2.0 to 1.0
MIN_CANDLES_REQUIRED = PERIOD # Need at least 'PERIOD' candles for SMA/STDDEV calculations


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure there are enough warm candles for the Bollinger Band calculation
        if len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue
        
        # Ensure there is hot data (current price) to compare against the bands
        if not pair_data.hot:
            continue

        # Extract the closing prices for the last 'PERIOD' warm candles
        closes_for_calc = [c.close for c in pair_data.warm[-PERIOD:]]

        # Calculate Simple Moving Average (SMA)
        mean = statistics.mean(closes_for_calc)

        # Calculate Standard Deviation (STDDEV)
        # statistics.stdev requires at least 2 data points.
        # Since PERIOD is 20, this condition is met if MIN_CANDLES_REQUIRED is also 20.
        # However, if all values are the same, stdev will be 0, which is handled below.
        if len(closes_for_calc) < 2: # Ensure enough data for stdev calculation in edge cases
             continue
        std = statistics.stdev(closes_for_calc)

        # Handle the case where standard deviation is zero (all prices are the same),
        # as bands would collapse and signals would be meaningless.
        if std == 0:
            continue

        # Calculate Upper and Lower Bollinger Bands
        upper_band = mean + (std * STD_DEV_MULTIPLIER)
        lower_band = mean - (std * STD_DEV_MULTIPLIER)

        # Get the current price and timestamp from the latest tick data
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # Generate Signals based on price breaching the bands
        if current_price < lower_band:
            # Price fell below the lower band, expect reversion upwards (Buy signal)
            signals.append(BuySignal(
                pair=pair,
                timestamp=ts,
                price=current_price,
                rule_id="rule_02_bollinger_band_tight_reversion"
            ))
        elif current_price > upper_band:
            # Price rose above the upper band, expect reversion downwards (Sell signal)
            signals.append(SellSignal(
                pair=pair,
                timestamp=ts,
                price=current_price,
                rule_id="rule_02_bollinger_band_tight_reversion"
            ))

    return signals