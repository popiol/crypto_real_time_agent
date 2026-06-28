"""Rule 08 — Momentum: rate-of-change (ROC) regime change.

Computes ROC at two time scales and emits a buy signal when both conditions hold:
  1. Short-term ROC (hot tier, last SHORT_WINDOW ticks) is positive AND
     accelerating (current window > previous window).
  2. Medium-term ROC (warm tier, last MEDIUM_WINDOW hours) has recently turned
     from negative to positive — a momentum regime change.

The medium-term regime change is detected by checking that the current
MEDIUM_WINDOW-period ROC is positive while at least one of the prior
REGIME_LOOKBACK measurements was negative or zero.
"""

from __future__ import annotations

from src.agent.models import BuySignal, PairData

RULE_ID = "roc_momentum"

SHORT_WINDOW = 5                                          # ticks
MEDIUM_WINDOW = 6                                         # hours
REGIME_LOOKBACK = 3                                       # prior medium ROC values to check
MIN_TICKS = 2 * SHORT_WINDOW + 1                          # need two full short windows
MIN_WARM_CANDLES = MEDIUM_WINDOW + REGIME_LOOKBACK + 1   # = 10

MarketData = dict[str, PairData]


def rate_of_change_momentum(data: MarketData) -> list[BuySignal]:
    signals: list[BuySignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < MIN_TICKS or len(pair_data.warm) < MIN_WARM_CANDLES:
            continue

        prices = [t.last_price for t in ticks]

        # Short-term ROC: current window vs previous window
        roc_now = (prices[-1] - prices[-SHORT_WINDOW - 1]) / prices[-SHORT_WINDOW - 1]
        roc_prev = (prices[-SHORT_WINDOW - 1] - prices[-2 * SHORT_WINDOW - 1]) / prices[-2 * SHORT_WINDOW - 1]

        # Condition 1: positive and accelerating
        if roc_now <= 0 or roc_now <= roc_prev:
            continue

        closes = [c.close for c in pair_data.warm]

        # Medium-term ROC at current position
        roc_medium_now = (closes[-1] - closes[-MEDIUM_WINDOW - 1]) / closes[-MEDIUM_WINDOW - 1]

        # Condition 2a: medium-term momentum is now positive
        if roc_medium_now <= 0:
            continue

        # Condition 2b: it was recently negative (regime change, not a sustained bull run)
        prior_rocs = [
            (closes[-i - 1] - closes[-i - 1 - MEDIUM_WINDOW]) / closes[-i - 1 - MEDIUM_WINDOW]
            for i in range(1, REGIME_LOOKBACK + 1)
        ]
        if not any(r <= 0 for r in prior_rocs):
            continue

        signals.append(
            BuySignal(
                pair=pair,
                rule_id=RULE_ID,
                timestamp=ticks[-1].polled_at,
                price=prices[-1],
            )
        )

    return signals
