from __future__ import annotations

import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# --- Constants ---
# Bollinger Band period for the primary timeframe (hourly candles).
# Adjusted from 20 to 10 to ensure enough data points are available within the
# typical 24 warm candles when a HigherTimeframeFactor of 2 is applied.
BB_PERIOD_PRIMARY = 10
BB_STDDEV_PRIMARY = 2.0

# Factor for the higher timeframe (e.g., 2 for 2x, 3 for 3x the primary timeframe).
# If primary is 1-hour, factor 2 means 2-hour candles.
HIGHER_TIMEFRAME_FACTOR = 2

# Minimum number of warm (hourly) candles required.
# This ensures enough data for:
# 1. Primary timeframe BB calculation (BB_PERIOD_PRIMARY candles).
# 2. Higher timeframe BB calculation (BB_PERIOD_PRIMARY * HIGHER_TIMEFRAME_FACTOR warm candles
#    to form BB_PERIOD_PRIMARY higher-timeframe candles).
# 3. Confirmation logic (at least 2 * HIGHER_TIMEFRAME_FACTOR warm candles to form
#    the 'previous' and 'current' higher-timeframe candles).
MIN_WARM_CANDLES = max(
    BB_PERIOD_PRIMARY,
    BB_PERIOD_PRIMARY * HIGHER_TIMEFRAME_FACTOR,
    2 * HIGHER_TIMEFRAME_FACTOR
)


class AggregatedCandle:
    """A simplified candle object for Bollinger Band calculations,
    used for both primary and aggregated higher-timeframe data."""
    def __init__(self, open_price: float, high: float, low: float, close: float):
        self.open_price = open_price
        self.high = high
        self.low = low
        self.close = close


def aggregate_candles(warm_candles: list[WarmCandle], factor: int) -> list[AggregatedCandle]:
    """
    Aggregates a list of WarmCandle objects into higher timeframe candles.
    Each aggregated candle represents `factor` number of primary timeframe candles.
    Partial groups at the end are discarded.
    """
    if factor == 1:
        return [AggregatedCandle(c.open_price, c.high, c.low, c.close) for c in warm_candles]

    aggregated_candles = []
    current_group: list[WarmCandle] = []

    for candle in warm_candles:
        current_group.append(candle)
        if len(current_group) == factor:
            # Aggregate the group into a single higher-timeframe candle
            open_price = current_group[0].open_price
            high = max(c.high for c in current_group)
            low = min(c.low for c in current_group)
            close = current_group[-1].close  # Close of the last candle in the group
            aggregated_candles.append(AggregatedCandle(open_price, high, low, close))
            current_group = []  # Reset for the next group
            
    return aggregated_candles


def calculate_bollinger_bands(
    candles: list[AggregatedCandle],
    period: int,
    std_dev_factor: float
) -> tuple[float, float, float] | None:
    """
    Calculates Bollinger Bands (mean, upper, lower) for a list of candles.
    Uses the `period` most recent candles from the provided list.
    Returns (mean, upper_band, lower_band) or None if not enough data.
    """
    if len(candles) < period:
        return None  # Not enough data for the specified period

    # Use the 'period' most recent candles for calculation
    relevant_closes = [c.close for c in candles[-period:]]
    
    mean = statistics.mean(relevant_closes)
    
    # Standard deviation requires at least two data points.
    # If all values are the same, stdev will be 0.
    if len(relevant_closes) < 2:
        std = 0.0
    else:
        try:
            std = statistics.stdev(relevant_closes)
        except statistics.StatisticsError: # Handles cases where all values are identical
            std = 0.0

    if std == 0:
        # If standard deviation is zero, bands collapse to the mean
        return mean, mean, mean

    upper_band = mean + std_dev_factor * std
    lower_band = mean - std_dev_factor * std
    return mean, upper_band, lower_band


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates Buy/Sell signals based on Bollinger Band Reversion with Multi-Timeframe Confirmation.

    A Buy signal is generated when:
    1. The current price drops below the lower Bollinger Band on the primary (hourly) timeframe.
    2. Simultaneously, on a higher timeframe, the previous candle closed below its lower Bollinger Band,
       and the current higher timeframe candle has closed back inside (or above) its lower Bollinger Band.

    A Sell signal is generated for the inverse conditions:
    1. The current price rises above the upper Bollinger Band on the primary (hourly) timeframe.
    2. Simultaneously, on a higher timeframe, the previous candle closed above its upper Bollinger Band,
       and the current higher timeframe candle has closed back inside (or below) its upper Bollinger Band.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm (hourly) candles are available for all calculations
        if not pair_data.warm or len(pair_data.warm) < MIN_WARM_CANDLES:
            continue
        
        # We need at least one tick in 'hot' for current price and timestamp
        if not pair_data.hot:
            continue

        # --- Primary Timeframe Bollinger Band Calculation ---
        # Convert WarmCandle list to AggregatedCandle list for consistency with BB function
        primary_candles_for_bb = [
            AggregatedCandle(c.open_price, c.high, c.low, c.close) for c in pair_data.warm
        ]
        bb_primary_results = calculate_bollinger_bands(
            primary_candles_for_bb,
            BB_PERIOD_PRIMARY,
            BB_STDDEV_PRIMARY
        )

        if bb_primary_results is None:
            continue  # Not enough data for primary BB

        mean_primary, upper_bb_primary, lower_bb_primary = bb_primary_results
        
        # Current price is from the latest tick, not the latest candle close
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # --- Higher Timeframe Bollinger Band Calculation and Reversal Confirmation ---
        higher_tf_candles = aggregate_candles(pair_data.warm, HIGHER_TIMEFRAME_FACTOR)

        # Need at least two higher timeframe candles for the 'previous' and 'current' confirmation
        if len(higher_tf_candles) < 2:
            continue

        current_higher_tf_candle = higher_tf_candles[-1]
        previous_higher_tf_candle = higher_tf_candles[-2]

        # Calculate Bollinger Bands for the higher timeframe using the same parameters
        higher_tf_bb_results = calculate_bollinger_bands(
            higher_tf_candles,
            BB_PERIOD_PRIMARY, # As per pseudocode, use same BB period for higher TF
            BB_STDDEV_PRIMARY
        )

        if higher_tf_bb_results is None:
            continue  # Not enough data for higher TF BB

        _, upper_bb_higher_tf, lower_bb_higher_tf = higher_tf_bb_results

        # --- Buy Signal Condition ---
        # 1. Primary timeframe: Current price is below the lower Bollinger Band.
        # 2. Higher timeframe confirmation: The price was below its lower BB on the previous
        #    higher-timeframe candle and has now closed back inside (or above) it.
        if current_price < lower_bb_primary:
            if (previous_higher_tf_candle.close < lower_bb_higher_tf and
                    current_higher_tf_candle.close >= lower_bb_higher_tf):
                signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))

        # --- Sell Signal Condition ---
        # 1. Primary timeframe: Current price is above the upper Bollinger Band.
        # 2. Higher timeframe confirmation: The price was above its upper BB on the previous
        #    higher-timeframe candle and has now closed back inside (or below) it.
        elif current_price > upper_bb_primary:
            if (previous_higher_tf_candle.close > upper_bb_higher_tf and
                    current_higher_tf_candle.close <= upper_bb_higher_tf):
                signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals