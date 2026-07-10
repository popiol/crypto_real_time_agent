from __future__ import annotations

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick

# Parameters
IMBALANCE_LOOKBACK_WINDOW = 5     # Number of recent imbalance observations to consider
IMBALANCE_THRESHOLD = 0.3         # Absolute value threshold for imbalance
CONSISTENCY_PERCENTAGE = 0.7      # Minimum percentage (0.0-1.0) of recent imbalances that must meet the threshold

# Minimum number of ticks required for the rule to have enough data.
# This must be at least IMBALANCE_LOOKBACK_WINDOW.
MIN_TICKS = IMBALANCE_LOOKBACK_WINDOW


def _imbalance(tick: Tick) -> float | None:
    """Return the imbalance ratio for one tick, or None if volumes are zero."""
    if tick.order_book is not None:
        bid_vol = sum(lvl.volume for lvl in tick.order_book.bids)
        ask_vol = sum(lvl.volume for lvl in tick.order_book.asks)
    else:
        bid_vol = tick.bid_volume
        ask_vol = tick.ask_volume

    total = bid_vol + ask_vol
    if total == 0:
        return None
    return (bid_vol - ask_vol) / total


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        
        # Ensure we have enough recent ticks for the lookback window.
        # If IMBALANCE_LOOKBACK_WINDOW is 0, MIN_TICKS will be 0, allowing processing
        # as long as there are any ticks.
        if len(ticks) < MIN_TICKS:
            continue

        # Get the most recent ticks for the lookback window.
        recent_ticks = ticks[-IMBALANCE_LOOKBACK_WINDOW:]
        
        # This check confirms that the slice actually yielded enough ticks,
        # which is important if IMBALANCE_LOOKBACK_WINDOW could be larger than len(ticks).
        # Given MIN_TICKS = IMBALANCE_LOOKBACK_WINDOW, this is mostly defensive.
        if len(recent_ticks) < IMBALANCE_LOOKBACK_WINDOW:
            continue 

        # Calculate imbalances for the recent ticks.
        recent_obis = [_imbalance(t) for t in recent_ticks]

        # If any imbalance could not be calculated (e.g., zero total volume), skip this pair.
        if any(v is None for v in recent_obis):
            continue

        # At this point, all elements in recent_obis are guaranteed to be floats.
        recent_obis_float: list[float] = [v for v in recent_obis if v is not None] # type: ignore[misc]

        # Handle the edge case where IMBALANCE_LOOKBACK_WINDOW is 0.
        # While MIN_TICKS logic should prevent this, it's a safe guard against division by zero.
        if IMBALANCE_LOOKBACK_WINDOW == 0:
            continue

        # Count how many recent OBIs meet the positive threshold.
        positive_count = sum(1 for obi in recent_obis_float if obi > IMBALANCE_THRESHOLD)

        # Count how many recent OBIs meet the negative threshold.
        negative_count = sum(1 for obi in recent_obis_float if obi < -IMBALANCE_THRESHOLD)

        # Calculate the actual percentage of consistent imbalances.
        positive_percentage = positive_count / IMBALANCE_LOOKBACK_WINDOW
        negative_percentage = negative_count / IMBALANCE_LOOKBACK_WINDOW

        ts = ticks[-1].polled_at
        price = ticks[-1].last_price

        # Check for Buy signal.
        if positive_percentage >= CONSISTENCY_PERCENTAGE:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=price))
        # Check for Sell signal. Using 'elif' ensures only one signal (buy or sell) is generated
        # per tick, prioritizing buying pressure if both conditions were somehow met
        # (which is unlikely with absolute thresholds).
        elif negative_percentage >= CONSISTENCY_PERCENTAGE:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=price))

    return signals