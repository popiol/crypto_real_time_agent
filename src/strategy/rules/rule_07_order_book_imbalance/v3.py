# Rule 4e5fcdd7-3e2b-4c99-81d0-98b09582ba39 — Adaptive Thresholds for Order Book Imbalance Signal Generation.
from __future__ import annotations
import numpy as np # For efficient rolling mean and std dev
from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, OrderBook, BaseModel, datetime, Field


# Configuration constants for the rule
IMBALANCE_CALC_LOOKBACK_PERIOD = 10  # Lookback for current average imbalance (short-term)
STD_DEV_LOOKBACK_PERIOD = 200        # Lookback for rolling std dev of average imbalance (long-term)
THRESHOLD_MULTIPLIER = 1.5           # Multiplier for std dev to set signal threshold

# Minimum total ticks required in hot data for a pair.
# This ensures we have enough data to calculate:
# 1. The `current_avg_imbalance` for the most recent point (requires IMBALANCE_CALC_LOOKBACK_PERIOD raw ticks).
# 2. A history of `current_avg_imbalance` values long enough (STD_DEV_LOOKBACK_PERIOD)
#    to compute its rolling mean and standard deviation.
# The total number of raw ticks needed is:
# (number of avg_imbalances needed for std dev) + (lookback for one avg_imbalance) - 1
MIN_TICKS = STD_DEV_LOOKBACK_PERIOD + IMBALANCE_CALC_LOOKBACK_PERIOD - 1


def _imbalance(tick: Tick) -> float:
    """
    Calculate the order book imbalance ratio for a single tick.
    Imbalance = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume).
    Returns 0.0 if total volume is zero to avoid division by zero and
    align with the pseudocode's handling of zero volume.
    """
    if tick.order_book is not None:
        bid_vol = sum(lvl.volume for lvl in tick.order_book.bids)
        ask_vol = sum(lvl.volume for lvl in tick.order_book.asks)
    else:
        # Fallback to Ticker's best bid/ask volumes if order book is not available
        bid_vol = tick.bid_volume
        ask_vol = tick.ask_volume

    total_volume = bid_vol + ask_vol
    if total_volume == 0:
        return 0.0  # Return 0.0 for no volume to signify no pressure
    return (bid_vol - ask_vol) / total_volume


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on adaptive thresholds for order book imbalance.
    Thresholds are calculated dynamically using a multiple of the rolling standard
    deviation of the average order book imbalance. This aims to make the rule
    more sensitive to relative shifts in imbalance, rather than fixed absolute values.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot

        # Ensure there's enough historical data overall for all calculations
        if len(ticks) < MIN_TICKS:
            continue

        # Step 1: Calculate raw instantaneous imbalance for all available ticks
        raw_imbalances = np.array([_imbalance(t) for t in ticks])

        # Step 2: Calculate the historical series of `current_avg_imbalance`
        # This is a rolling mean of raw imbalances over IMBALANCE_CALC_LOOKBACK_PERIOD.
        # The result will have length `len(raw_imbalances) - IMBALANCE_CALC_LOOKBACK_PERIOD + 1`.
        # We ensure `len(raw_imbalances)` is sufficient via the `MIN_TICKS` check.
        
        # Manually compute rolling mean to create `historical_avg_imbalance_series`
        # This series will contain `current_avg_imbalance` values over time.
        avg_imbalances_history_list = []
        for i in range(IMBALANCE_CALC_LOOKBACK_PERIOD - 1, len(raw_imbalances)):
            window = raw_imbalances[i - IMBALANCE_CALC_LOOKBACK_PERIOD + 1 : i + 1]
            avg_imbalances_history_list.append(np.mean(window))
        
        avg_imbalances_history = np.array(avg_imbalances_history_list)

        # The length of `avg_imbalances_history` must be at least `STD_DEV_LOOKBACK_PERIOD`
        # for the next step. This is guaranteed by `MIN_TICKS`.
        # `len(avg_imbalances_history)` = `len(raw_imbalances) - IMBALANCE_CALC_LOOKBACK_PERIOD + 1`
        # `MIN_TICKS - IMBALANCE_CALC_LOOKBACK_PERIOD + 1` = `(STD_DEV_LOOKBACK_PERIOD + IMBALANCE_CALC_LOOKBACK_PERIOD - 1) - IMBALANCE_CALC_LOOKBACK_PERIOD + 1`
        # = `STD_DEV_LOOKBACK_PERIOD`. So, `avg_imbalances_history` will always have at least `STD_DEV_LOOKBACK_PERIOD` elements.

        # Step 3: Calculate rolling mean and standard deviation of the historical average imbalances
        # We use the most recent `STD_DEV_LOOKBACK_PERIOD` values from `avg_imbalances_history`.
        relevant_avg_imbalances = avg_imbalances_history[-STD_DEV_LOOKBACK_PERIOD:]

        rolling_mean_of_avg_imbalance = np.mean(relevant_avg_imbalances)
        rolling_std_dev_of_avg_imbalance = np.std(relevant_avg_imbalances)

        # If standard deviation is zero (all relevant historical average imbalances were identical),
        # adaptive thresholds would be fixed at the mean. In this edge case, we skip signal generation
        # as there's no dynamic range to adapt to.
        if rolling_std_dev_of_avg_imbalance == 0:
            continue

        # Step 4: Calculate adaptive thresholds
        buy_threshold = rolling_mean_of_avg_imbalance + (THRESHOLD_MULTIPLIER * rolling_std_dev_of_avg_imbalance)
        sell_threshold = rolling_mean_of_avg_imbalance - (THRESHOLD_MULTIPLIER * rolling_std_dev_of_avg_imbalance)

        # Step 5: Get the current average imbalance (the most recent one calculated)
        current_avg_imbalance = avg_imbalances_history[-1]

        # Get the timestamp and price from the most recent tick for the signal
        latest_tick = ticks[-1]
        timestamp = latest_tick.polled_at
        price = latest_tick.last_price

        # Generate signals based on the current average imbalance against adaptive thresholds
        if current_avg_imbalance > buy_threshold:
            signals.append(BuySignal(pair=pair, timestamp=timestamp, price=price))
        elif current_avg_imbalance < sell_threshold:
            signals.append(SellSignal(pair=pair, timestamp=timestamp, price=price))

    return signals