"""Rule 07 — Market microstructure: order book imbalance.

Computes the bid/ask volume imbalance ratio at each hot-tier tick:

    imbalance = (bid_vol − ask_vol) / (bid_vol + ask_vol)  ∈ [−1, 1]

Uses the top-5 order book from the Depth endpoint when available, falling back
to the single best bid/ask volumes from the Ticker. Emits a buy signal when
the imbalance exceeds IMBALANCE_THRESHOLD and has been sustained for at least
MIN_SUSTAINED_TICKS consecutive ticks, indicating persistent buying pressure.
"""

from __future__ import annotations

from src.agent.models import BuySignal, PairData, Tick

RULE_ID = "order_book_imbalance"

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


def order_book_imbalance(data: MarketData) -> list[BuySignal]:
    signals: list[BuySignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < MIN_TICKS:
            continue

        recent = ticks[-MIN_SUSTAINED_TICKS:]
        if len(recent) < MIN_SUSTAINED_TICKS:
            continue

        if all(
            (v := _imbalance(t)) is not None and v > IMBALANCE_THRESHOLD
            for t in recent
        ):
            signals.append(
                BuySignal(
                    pair=pair,
                    rule_id=RULE_ID,
                    timestamp=ticks[-1].polled_at,
                    price=ticks[-1].last_price,
                )
            )

    return signals
