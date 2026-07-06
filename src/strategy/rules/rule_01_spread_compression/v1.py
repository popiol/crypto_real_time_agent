"""Rule 01 — Spread compression / expansion.

Buy signal:  spread drops more than THRESHOLD below rolling baseline
             → tight market conditions, can precede upward moves.
Sell signal: spread rises more than THRESHOLD above rolling baseline
             → deteriorating liquidity, exit condition.
"""

from __future__ import annotations

import statistics

from src.agent.models import BuySignal, PairData, SellSignal

RULE_ID = "rule_01_spread_compression_v1"

MIN_TICKS = 10
THRESHOLD = 0.30

MarketData = dict[str, PairData]


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < MIN_TICKS:
            continue

        spreads = [t.spread_rel for t in ticks]
        baseline = statistics.mean(spreads[:-1])
        current = spreads[-1]

        if baseline <= 0:
            continue

        ts = ticks[-1].polled_at
        price = ticks[-1].last_price

        if current < baseline * (1 - THRESHOLD):
            signals.append(BuySignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price))
        elif current > baseline * (1 + THRESHOLD):
            signals.append(SellSignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price))

    return signals
