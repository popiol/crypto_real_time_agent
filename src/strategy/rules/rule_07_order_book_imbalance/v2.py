from __future__ import annotations

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick


# Constants
IMBALANCE_THRESHOLD = 0.3
LOOKBACK_WINDOW_TICKS = 5  # The number of recent ticks to average imbalance over


def _imbalance(tick: Tick) -> float | None:
    """
    Return the imbalance ratio for one tick, or None if total volume is zero.
    This calculation is consistent with rule_07_order_book_imbalance_v1,
    using (bid_volume - ask_volume) / (bid_volume + ask_volume).
    """
    if tick.order_book is not None:
        bid_vol = sum(lvl.volume for lvl in tick.order_book.bids)
        ask_vol = sum(lvl.volume for lvl in tick.order_book.asks)
    else:
        # Fallback to Ticker data if OrderBook is not available
        bid_vol = tick.bid_volume
        ask_vol = tick.ask_volume

    total = bid_vol + ask_vol
    if total == 0:
        return None  # Cannot calculate imbalance if total volume is zero
    return (bid_vol - ask_vol) / total


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on the average order book imbalance
    over a specified lookback window.

    A Buy signal is generated if the average imbalance is above a positive threshold.
    A Sell signal is generated if the average imbalance is below a negative threshold.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot

        # Ensure there are enough ticks to form the lookback window
        if len(ticks) < LOOKBACK_WINDOW_TICKS:
            continue

        # Get the recent ticks for the lookback window
        recent_ticks = ticks[-LOOKBACK_WINDOW_TICKS:]

        # Calculate imbalances for each tick in the recent window
        imbalances: list[float] = []
        for tick in recent_ticks:
            imbal = _imbalance(tick)
            if imbal is None:
                # If any imbalance cannot be calculated (e.g., zero total volume),
                # this lookback window is invalid.
                imbalances = []  # Clear any partial list
                break
            imbalances.append(imbal)

        # If imbalances list is empty, it means an invalid imbalance was found
        # or there were not enough valid ticks.
        if not imbalances:
            continue

        # Calculate the average imbalance over the lookback window
        average_imbalance = sum(imbalances) / LOOKBACK_WINDOW_TICKS

        # Get the timestamp and price from the most recent tick
        ts = ticks[-1].polled_at
        price = ticks[-1].last_price

        # Generate signals based on the average imbalance compared to the threshold
        if average_imbalance > IMBALANCE_THRESHOLD:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=price))
        elif average_imbalance < -IMBALANCE_THRESHOLD:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=price))

    return signals