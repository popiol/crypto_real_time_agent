"""Rule 31027b91-449f-4175-a374-1b629e2f63ef — Modify Order Book Imbalance v1: Average Imbalance Threshold."""
from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, OrderBook, BaseModel, datetime, Field


# Configuration constants for the rule
MIN_TICKS = 10                  # Minimum total ticks required in hot data for a pair
IMBALANCE_THRESHOLD = 0.3       # The threshold for the average order book imbalance
LOOKBACK_PERIOD = 5             # The number of recent ticks to average the imbalance over


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
    Generates trading signals based on the average order book imbalance
    over a recent lookback period.

    A BuySignal is generated if the average imbalance is significantly positive
    (indicating sustained buying pressure).
    A SellSignal is generated if the average imbalance is significantly negative
    (indicating sustained selling pressure).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot

        # Ensure there's enough historical data overall
        if len(ticks) < MIN_TICKS:
            continue

        # Extract the recent ticks for the lookback period
        recent_ticks = ticks[-LOOKBACK_PERIOD:]

        # Ensure there are enough ticks within the specified lookback period
        if len(recent_ticks) < LOOKBACK_PERIOD:
            continue

        # Calculate imbalance for each tick in the recent period
        imbalances = [_imbalance(t) for t in recent_ticks]

        # If for some reason the list of imbalances is empty (should not happen
        # with the above checks, but as a safeguard)
        if not imbalances:
            continue

        # Calculate the average imbalance over the lookback period
        average_imbalance = statistics.mean(imbalances)

        # Get the timestamp and price from the most recent tick for the signal
        latest_tick = ticks[-1]
        timestamp = latest_tick.polled_at
        price = latest_tick.last_price

        # Generate signals based on the average imbalance
        if average_imbalance > IMBALANCE_THRESHOLD:
            signals.append(BuySignal(pair=pair, timestamp=timestamp, price=price))
        elif average_imbalance < -IMBALANCE_THRESHOLD:
            signals.append(SellSignal(pair=pair, timestamp=timestamp, price=price))

    return signals