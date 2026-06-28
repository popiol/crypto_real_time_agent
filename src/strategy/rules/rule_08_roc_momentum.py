"""Rule 08 — Momentum: rate-of-change (ROC) regime change.

Computes ROC at two time scales.

Buy signal:
  1. Short-term ROC positive AND accelerating (current > previous window).
  2. Medium-term ROC just turned positive from negative (regime change upward).

Sell signal:
  1. Short-term ROC negative AND decelerating (current < previous window).
  2. Medium-term ROC just turned negative from positive (regime change downward).

The regime change is detected by checking that at least one of the prior
REGIME_LOOKBACK medium-term ROC values had the opposite sign.
"""

from __future__ import annotations

from src.agent.models import BuySignal, PairData, SellSignal

RULE_ID = "roc_momentum"

SHORT_WINDOW = 5                                          # ticks
MEDIUM_WINDOW = 6                                         # hours
REGIME_LOOKBACK = 3                                       # prior medium ROC values to check
MIN_TICKS = 2 * SHORT_WINDOW + 1                          # need two full short windows
MIN_WARM_CANDLES = MEDIUM_WINDOW + REGIME_LOOKBACK + 1   # = 10

MarketData = dict[str, PairData]


def rate_of_change_momentum(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < MIN_TICKS or len(pair_data.warm) < MIN_WARM_CANDLES:
            continue

        prices = [t.last_price for t in ticks]
        roc_now = (prices[-1] - prices[-SHORT_WINDOW - 1]) / prices[-SHORT_WINDOW - 1]
        roc_prev = (prices[-SHORT_WINDOW - 1] - prices[-2 * SHORT_WINDOW - 1]) / prices[-2 * SHORT_WINDOW - 1]

        closes = [c.close for c in pair_data.warm]
        roc_medium_now = (closes[-1] - closes[-MEDIUM_WINDOW - 1]) / closes[-MEDIUM_WINDOW - 1]
        prior_rocs = [
            (closes[-i - 1] - closes[-i - 1 - MEDIUM_WINDOW]) / closes[-i - 1 - MEDIUM_WINDOW]
            for i in range(1, REGIME_LOOKBACK + 1)
        ]

        ts = ticks[-1].polled_at
        price = prices[-1]

        if roc_now > 0 and roc_now > roc_prev and roc_medium_now > 0 and any(r <= 0 for r in prior_rocs):
            signals.append(BuySignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price))
        elif roc_now < 0 and roc_now < roc_prev and roc_medium_now < 0 and any(r >= 0 for r in prior_rocs):
            signals.append(SellSignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price))

    return signals
