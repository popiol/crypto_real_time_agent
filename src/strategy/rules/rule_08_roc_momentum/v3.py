"""Rule XX — ROC Crossover Momentum Signal."""
from __future__ import annotations
from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick
from datetime import datetime

# Parameters for Rate of Change (ROC) periods
# These are mapped from the original rule's SHORT_ROC_PERIOD and MEDIUM_ROC_PERIOD
# to represent the 'fast' and 'slow' ROCs respectively.
FAST_ROC_PERIOD = 5  # e.g., 5 periods for fast-term ROC
SLOW_ROC_PERIOD = 20 # e.g., 20 periods for slow-term ROC

# Minimum data requirements for calculations:
# To detect a crossover, we need to calculate ROCs for both the current and the previous time step.
# The longest period is SLOW_ROC_PERIOD.
# For current ROCs, we need `SLOW_ROC_PERIOD + 1` data points (index 0 to SLOW_ROC_PERIOD).
# For previous ROCs, we need `SLOW_ROC_PERIOD + 1` data points ending one step earlier,
# which means we need `SLOW_ROC_PERIOD + 2` data points in total (index 0 to SLOW_ROC_PERIOD + 1).
MIN_TICKS = max(FAST_ROC_PERIOD, SLOW_ROC_PERIOD) + 2


def _calculate_roc(prices_data: list[Tick], period: int, current_idx: int) -> float:
    """
    Calculates the Rate of Change (ROC) for a given period,
    ending at the specified current_idx in the prices_data list.

    Args:
        prices_data: A list of Tick objects containing price information.
        period: The number of periods to look back for the ROC calculation.
        current_idx: The index of the current tick in prices_data.

    Returns:
        The calculated ROC as a float. Returns float('nan') if there's insufficient
        data or if the historical price is zero (to avoid division by zero).
    """
    # Ensure there's enough historical data for the requested period
    if current_idx < period:
        return float('nan')

    current_price = prices_data[current_idx].last_price
    previous_price = prices_data[current_idx - period].last_price

    # Avoid division by zero. A zero historical price makes ROC undefined.
    if previous_price == 0:
        return float('nan')

    return (current_price - previous_price) / previous_price


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on the crossover of a fast Rate of Change (ROC)
    over a slow Rate of Change.

    A Buy signal is emitted when the fast ROC crosses above the slow ROC.
    A Sell signal is emitted when the fast ROC crosses below the slow ROC.

    Args:
        data: A MarketData object containing tick and candle data for various pairs.

    Returns:
        A list of BuySignal or SellSignal objects.
    """
    signals: list[BuySignal | SellSignal] = []
    rule_id = "roc_crossover_momentum"

    for pair, pair_data in data.items():
        ticks = pair_data.hot

        # Ensure we have enough tick data to calculate current and previous ROCs
        if len(ticks) < MIN_TICKS:
            continue

        # Calculate ROCs for the current time step (last tick)
        current_tick_idx = len(ticks) - 1
        fast_roc_current = _calculate_roc(ticks, FAST_ROC_PERIOD, current_tick_idx)
        slow_roc_current = _calculate_roc(ticks, SLOW_ROC_PERIOD, current_tick_idx)

        # Calculate ROCs for the previous time step (second to last tick)
        previous_tick_idx = len(ticks) - 2
        fast_roc_previous = _calculate_roc(ticks, FAST_ROC_PERIOD, previous_tick_idx)
        slow_roc_previous = _calculate_roc(ticks, SLOW_ROC_PERIOD, previous_tick_idx)

        # Skip signal generation if any ROC calculation resulted in NaN (insufficient data or zero price)
        if any(map(lambda x: x != x, [fast_roc_current, slow_roc_current, fast_roc_previous, slow_roc_previous])):
            continue

        # Get the timestamp and price for the current signal
        ts: datetime = ticks[-1].polled_at
        price: float = ticks[-1].last_price

        # --- Signal Generation Logic ---
        # Buy signal: Fast ROC crosses above Slow ROC
        # This means fast_roc_current is greater than slow_roc_current,
        # AND fast_roc_previous was less than or equal to slow_roc_previous.
        if fast_roc_current > slow_roc_current and fast_roc_previous <= slow_roc_previous:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=price, rule_id=rule_id))
        # Sell signal: Fast ROC crosses below Slow ROC
        # This means fast_roc_current is less than slow_roc_current,
        # AND fast_roc_previous was greater than or equal to slow_roc_previous.
        elif fast_roc_current < slow_roc_current and fast_roc_previous >= slow_roc_previous:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=price, rule_id=rule_id))

    return signals