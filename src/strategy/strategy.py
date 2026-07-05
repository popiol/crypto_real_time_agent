"""Strategy engine — signal detection.

find_signals() is the only public entry point. It calls every registered
rule function and merges the results.

To add a rule: create src/strategy/rules/<rule_name>/v1.py, import its
function below, and append it to ACTIVE_RULES.
To add a version: create v2.py in the existing rule's folder, import it,
and append it alongside the prior version while it is being evaluated.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from src.agent.models import BuySignal, PairData, SellSignal
from src.strategy.rules.rule_01_spread_compression.v1 import spread_compression_spike
from src.strategy.rules.rule_02_bollinger_band.v1 import bollinger_band_lower_touch
from src.strategy.rules.rule_03_ou_spread.v1 import ou_spread_compression
from src.strategy.rules.rule_06_kalman_velocity.v1 import kalman_velocity_reversal
from src.strategy.rules.rule_07_order_book_imbalance.v1 import order_book_imbalance
from src.strategy.rules.rule_08_roc_momentum.v1 import rate_of_change_momentum
from src.strategy.rules.rule_10_cnn_forecast.v1 import cnn_price_forecast
from src.strategy.rules.rule_11_dqn_agent.v1 import dqn_buy_signal

MarketData = dict[str, PairData]
Signal = BuySignal | SellSignal
RuleFn = Callable[[MarketData], Sequence[Signal]]

ACTIVE_RULES: list[RuleFn] = [
    spread_compression_spike,
    bollinger_band_lower_touch,
    ou_spread_compression,
    kalman_velocity_reversal,
    order_book_imbalance,
    rate_of_change_momentum,
    cnn_price_forecast,
    dqn_buy_signal,
]


def find_signals(data: MarketData) -> list[Signal]:
    """Run all active rules against the current market data and return signals."""
    signals: list[Signal] = []
    for rule_fn in ACTIVE_RULES:
        try:
            signals.extend(rule_fn(data))
        except Exception:  # noqa: BLE001
            logging.exception("Rule %s raised an exception", rule_fn.__name__)
    return signals
