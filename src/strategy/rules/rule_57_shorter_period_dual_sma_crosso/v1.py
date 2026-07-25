from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

# Rule specific constants
RULE_ID = "SMA-5-20-CROSSOVER"
FAST_SMA_PERIOD = 5
SLOW_SMA_PERIOD = 20

# We need at least SLOW_SMA_PERIOD candles to calculate the current slow SMA.
# To detect a crossover, we also need the previous SMA values, which means we need
# SLOW_SMA_PERIOD + 1 candles in total to get both current and previous values for the slow SMA.
MIN_CANDLES_REQUIRED = SLOW_SMA_PERIOD + 1

def _calculate_sma(prices: list[float], period: int) -> float:
    """Calculates the Simple Moving Average for a given period."""
    if len(prices) < period:
        return np.nan
    return float(np.mean(prices[-period:]))

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure we have enough data for both SMAs and crossover detection
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract close prices from the warm candles
        close_prices = [candle.close for candle in warm_candles]

        # Calculate current SMAs using the most recent candles
        sma_5_current = _calculate_sma(close_prices, FAST_SMA_PERIOD)
        sma_20_current = _calculate_sma(close_prices, SLOW_SMA_PERIOD)

        # Calculate previous SMAs using data up to the second-to-last candle
        # This slice effectively removes the very latest candle, giving us the "previous" state
        previous_close_prices = close_prices[:-1]

        # We must ensure there's enough data for previous SMAs as well.
        # This check is technically redundant if MIN_CANDLES_REQUIRED is correctly set,
        # but provides an additional safeguard.
        if len(previous_close_prices) < SLOW_SMA_PERIOD:
            continue

        sma_5_previous = _calculate_sma(previous_close_prices, FAST_SMA_PERIOD)
        sma_20_previous = _calculate_sma(previous_close_prices, SLOW_SMA_PERIOD)

        # Skip if any SMA calculation resulted in NaN (due to insufficient data within helper, though guarded above)
        if np.isnan(sma_5_current) or np.isnan(sma_20_current) or \
           np.isnan(sma_5_previous) or np.isnan(sma_20_previous):
            continue

        latest_candle = warm_candles[-1]
        timestamp = latest_candle.hour
        price = latest_candle.close

        # Long Entry Signal: 5-period SMA crosses above 20-period SMA
        if sma_5_current > sma_20_current and sma_5_previous <= sma_20_previous:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=price,
                rule_id=RULE_ID
            ))
        # Short Entry Signal: 5-period SMA crosses below 20-period SMA
        elif sma_5_current < sma_20_current and sma_5_previous >= sma_20_previous:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=price,
                rule_id=RULE_ID
            ))

    return signals