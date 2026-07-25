from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

FAST_SMA_PERIOD = 10
SLOW_SMA_PERIOD = 30
RULE_ID = "SMA-Crossover-Baseline"

def calculate_sma(prices: list[float], period: int) -> float | None:
    """Calculates the Simple Moving Average for the given period."""
    if len(prices) < period:
        return None
    return statistics.mean(prices[-period:])

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Dual Moving Average Crossover Strategy.

    This rule initiates a long position when a fast-moving average (10-period SMA)
    crosses above a slow-moving average (30-period SMA) using hourly close prices.
    It initiates a short position when the fast SMA crosses below the slow SMA.
    Positions are closed upon the occurrence of the opposite crossover.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # To detect a crossover, we need to compare current and previous SMA values.
        # This requires enough candles to calculate the SLOW_SMA_PERIOD for both
        # the current and the immediately preceding period.
        # Thus, we need at least SLOW_SMA_PERIOD + 1 candles.
        # Given that data.warm holds at most 24 hourly candles, and SLOW_SMA_PERIOD is 30,
        # this condition (len(warm_candles) >= 31) will almost always fail,
        # leading to no signals being generated with the current data model constraints.
        required_candles_for_crossover = SLOW_SMA_PERIOD + 1
        if len(warm_candles) < required_candles_for_crossover:
            # Insufficient data to calculate both current and previous SMAs reliably for a crossover.
            continue

        # Extract close prices for the required lookback period.
        # We need prices up to the current candle (index -1) and the one before (index -2)
        # to calculate both current and previous SMAs.
        close_prices = [c.close for c in warm_candles]

        # Calculate SMAs for the current candle (using the most recent `period` candles)
        fast_sma_current = calculate_sma(close_prices, FAST_SMA_PERIOD)
        slow_sma_current = calculate_sma(close_prices, SLOW_SMA_PERIOD)

        # Calculate SMAs for the previous candle (using candles up to the second-to-last one)
        close_prices_previous_period = close_prices[:-1]
        fast_sma_previous = calculate_sma(close_prices_previous_period, FAST_SMA_PERIOD)
        slow_sma_previous = calculate_sma(close_prices_previous_period, SLOW_SMA_PERIOD)

        # If any SMA calculation returned None (due to insufficient data within the slice,
        # though this should be covered by required_candles_for_crossover check), skip.
        if any(sma is None for sma in [fast_sma_current, slow_sma_current, fast_sma_previous, slow_sma_previous]):
            continue

        # Get the latest candle's details for signal generation
        latest_candle = warm_candles[-1]
        timestamp = latest_candle.hour
        price = latest_candle.close

        # Crossover logic:
        # Long position: Fast SMA crosses above Slow SMA
        if (fast_sma_current > slow_sma_current and
                fast_sma_previous <= slow_sma_previous):
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence as it's a direct crossover signal
            ))
        # Short position: Fast SMA crosses below Slow SMA
        elif (fast_sma_current < slow_sma_current and
                fast_sma_previous >= slow_sma_previous):
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence as it's a direct crossover signal
            ))

    return signals