"""Strategy engine — signal detection.

find_signals() is the only public entry point. It calls every registered
rule function and merges the results.

To add a rule: create src/strategy/rules/rule_NN_<name>.py, then import its
function below and append it to ACTIVE_RULES.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from src.agent.models import BuySignal, PairData, SellSignal
from src.strategy.rules.rule_01_spread_compression import spread_compression_spike
from src.strategy.rules.rule_02_bollinger_band import bollinger_band_lower_touch
from src.strategy.rules.rule_03_ou_spread import ou_spread_compression
from src.strategy.rules.rule_04_arima_forecast import arima_price_forecast
from src.strategy.rules.rule_05_fft_cycle import fft_cycle_trough
from src.strategy.rules.rule_06_kalman_velocity import kalman_velocity_reversal
from src.strategy.rules.rule_07_order_book_imbalance import order_book_imbalance
from src.strategy.rules.rule_08_roc_momentum import rate_of_change_momentum
from src.strategy.rules.rule_09_markov_chain import markov_chain_transition
from src.strategy.rules.rule_10_cnn_forecast import cnn_price_forecast
from src.strategy.rules.rule_11_dqn_agent import dqn_buy_signal
from src.strategy.rules.rule_12_lead_lag import lead_lag_cross_asset

MarketData = dict[str, PairData]
Signal = BuySignal | SellSignal
RuleFn = Callable[[MarketData], Sequence[Signal]]

ACTIVE_RULES: list[RuleFn] = [
    spread_compression_spike,
    bollinger_band_lower_touch,
    ou_spread_compression,
    arima_price_forecast,
    fft_cycle_trough,
    kalman_velocity_reversal,
    order_book_imbalance,
    rate_of_change_momentum,
    markov_chain_transition,
    cnn_price_forecast,
    dqn_buy_signal,
    lead_lag_cross_asset,
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
