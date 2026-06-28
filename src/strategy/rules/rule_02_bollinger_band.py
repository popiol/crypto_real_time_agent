"""Rule 02 — Mean reversion: Bollinger Band touch.

Buy signal:  price below lower band (mean − K·σ) → downward overextension, expect reversion up.
Sell signal: price above upper band (mean + K·σ) → upward overextension, expect reversion down.
"""

from __future__ import annotations

import statistics

from src.agent.models import BuySignal, PairData, SellSignal

RULE_ID = "bollinger_band_lower_touch"

K = 2.0
MIN_CANDLES = 10

MarketData = dict[str, PairData]


def bollinger_band_lower_touch(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES:
            continue

        closes = [c.close for c in pair_data.warm]
        mean = statistics.mean(closes)
        std = statistics.stdev(closes)

        if std == 0:
            continue

        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        if current_price < mean - K * std:
            signals.append(BuySignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=current_price))
        elif current_price > mean + K * std:
            signals.append(SellSignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=current_price))

    return signals
