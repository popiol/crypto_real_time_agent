from __future__ import annotations
import numpy as np
from datetime import datetime

from src.agent.models import BuySignal, MarketData, SellSignal, ColdMonth, WarmCandle, Tick

# Rule specific parameters
SHORT_PERIOD = 10
LONG_PERIOD = 30

def calculate_sma(prices: list[float], period: int) -> float | np.nan:
    """
    Calculates the Simple Moving Average for the last 'period' prices.
    Returns np.nan if there isn't enough data.
    """
    if len(prices) < period:
        return np.nan
    return np.mean(prices[-period:])

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Retrieve cold month data and sort it chronologically to ensure correct SMA calculation
        cold_months = sorted(pair_data.cold, key=lambda m: m.month)

        # Extract avg_price from ColdMonth objects.
        # As per the pseudocode's `data.cold.close_prices` and the `ColdMonth` model structure,
        # `avg_price` is used as the representative price for monthly periods.
        prices = [m.avg_price for m in cold_months]

        # We need enough historical data to calculate both the current and the previous
        # value for the longer SMA. This means at least LONG_PERIOD + 1 data points.
        # For example, if LONG_PERIOD=30, we need 31 monthly prices:
        # 30 for the current SMA, and the preceding 30 for the previous SMA.
        if len(prices) < LONG_PERIOD + 1:
            continue

        # Get current tick data for signal timestamp and price.
        # This provides the most up-to-date market context for the signal.
        ticks = pair_data.hot
        if not ticks:
            # If no current market data, we cannot generate a signal with a valid price/timestamp.
            continue

        current_timestamp = ticks[-1].polled_at
        current_price = ticks[-1].last_price

        # Calculate current SMAs using the very latest data point
        current_short_sma = calculate_sma(prices, SHORT_PERIOD)
        current_long_sma = calculate_sma(prices, LONG_PERIOD)

        # Calculate previous SMAs using data up to the second-to-last data point.
        # `prices[:-1]` represents the data series excluding the latest month's price.
        previous_short_sma = calculate_sma(prices[:-1], SHORT_PERIOD)
        previous_long_sma = calculate_sma(prices[:-1], LONG_PERIOD)

        # If any SMA calculation failed (e.g., due to insufficient data for a specific period),
        # skip this pair. This handles cases where SHORT_PERIOD might be too large for `prices[:-1]`,
        # though with current parameters (10, 30), this is covered by the initial `LONG_PERIOD + 1` check.
        if np.isnan(current_short_sma) or np.isnan(current_long_sma) or \
           np.isnan(previous_short_sma) or np.isnan(previous_long_sma):
            continue

        # Check for buy signal: 10-period SMA crosses above 30-period SMA
        # This indicates increasing short-term momentum relative to the longer-term trend.
        if current_short_sma > current_long_sma and previous_short_sma <= previous_long_sma:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id="SMA-Cross-10-30"
            ))
        # Check for sell signal: 10-period SMA crosses below 30-period SMA
        # This suggests weakening short-term momentum and potential downtrend initiation.
        elif current_short_sma < current_long_sma and previous_short_sma >= previous_long_sma:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id="SMA-Cross-10-30"
            ))

    return signals