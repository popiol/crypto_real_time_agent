"""Rule 01 — Spread compression spike.

Emits a buy signal when the current bid/ask spread drops more than
COMPRESSION_THRESHOLD below its rolling baseline, indicating unusually
tight market conditions that can precede upward price moves.
"""

from __future__ import annotations

import statistics

from src.agent.models import BuySignal, PairData

RULE_ID = "spread_compression_spike"

# Minimum number of ticks required to establish a baseline
MIN_TICKS = 10

# Fraction below the baseline spread that triggers the signal (0.30 = 30%)
COMPRESSION_THRESHOLD = 0.30

MarketData = dict[str, PairData]


def spread_compression_spike(data: MarketData) -> list[BuySignal]:
    signals: list[BuySignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < MIN_TICKS:
            continue

        spreads = [t.spread_rel for t in ticks]
        baseline = statistics.mean(spreads[:-1])
        current = spreads[-1]

        if baseline > 0 and current < baseline * (1 - COMPRESSION_THRESHOLD):
            signals.append(
                BuySignal(
                    pair=pair,
                    rule_id=RULE_ID,
                    timestamp=ticks[-1].polled_at,
                    price=ticks[-1].last_price,
                )
            )

    return signals
