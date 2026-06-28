"""Rule 02 — Mean reversion: Bollinger Band lower touch.

Emits a buy signal when the current price drops below the lower Bollinger Band
(mean − K·σ) computed from the last 24 hourly close prices in the warm tier,
indicating an unusually large downward deviation that may revert.
"""

from __future__ import annotations

import statistics

from src.agent.models import BuySignal, PairData

RULE_ID = "bollinger_band_lower_touch"

# Band width multiplier (standard value is 2.0)
K = 2.0

# Minimum warm candles required for reliable statistics
MIN_CANDLES = 10

MarketData = dict[str, PairData]


def bollinger_band_lower_touch(data: MarketData) -> list[BuySignal]:
    signals: list[BuySignal] = []

    for pair, pair_data in data.items():
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES:
            continue

        closes = [c.close for c in pair_data.warm]
        mean = statistics.mean(closes)
        std = statistics.stdev(closes)

        if std == 0:
            continue

        lower_band = mean - K * std
        current_price = pair_data.hot[-1].last_price

        if current_price < lower_band:
            signals.append(
                BuySignal(
                    pair=pair,
                    rule_id=RULE_ID,
                    timestamp=pair_data.hot[-1].polled_at,
                    price=current_price,
                )
            )

    return signals
