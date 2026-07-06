"""Rule 07 — Market microstructure: order book imbalance.

Computes the bid/ask volume imbalance ratio at each hot-tier tick:

    imbalance = (bid_vol − ask_vol) / (bid_vol + ask_vol)  ∈ [−1, 1]

Uses the top-5 order book from the Depth endpoint when available, falling back
to the single best bid/ask volumes from the Ticker.

Buy signal:  imbalance > +IMBALANCE_THRESHOLD sustained for MIN_SUSTAINED_TICKS → buying pressure.
Sell signal: imbalance < −IMBALANCE_THRESHOLD sustained for MIN_SUSTAINED_TICKS → selling pressure.
"""

from __future__ import annotations

from src.agent.models import BuySignal, PairData, SellSignal, Tick

RULE_ID = "rule_07_order_book_imbalance_v1"

MIN_TICKS = 10
IMBALANCE_THRESHOLD = 0.3   # net buy-side fraction required
MIN_SUSTAINED_TICKS = 5     # consecutive ticks the imbalance must hold

MarketData = dict[str, PairData]


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
        if len(ticks) < MIN_TICKS:
            continue

        recent = ticks[-MIN_SUSTAINED_TICKS:]
        if len(recent) < MIN_SUSTAINED_TICKS:
            continue

        imbalances = [_imbalance(t) for t in recent]
        if any(v is None for v in imbalances):
            continue

        ts = ticks[-1].polled_at
        price = ticks[-1].last_price

        if all(v > IMBALANCE_THRESHOLD for v in imbalances):  # type: ignore[operator]
            signals.append(BuySignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price))
        elif all(v < -IMBALANCE_THRESHOLD for v in imbalances):  # type: ignore[operator]
            signals.append(SellSignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price))

    return signals
