"""Strategy engine — signal detection.

find_signals() is the only public entry point. It iterates ACTIVE_RULES and
calls signal(data) on each rule module.

To add a rule: create src/strategy/rules/<rule_name>/v1.py with a signal()
function, then add an import and entry to ACTIVE_RULES below.
To add a version: create v2.py alongside v1.py and add it to ACTIVE_RULES
while it is being evaluated against the prior version.
"""

from __future__ import annotations

import logging
from types import ModuleType

import src.strategy.rules.rule_01_spread_compression.v1 as rule_01_spread_compression_v1
import src.strategy.rules.rule_12_lead_lag.v3 as rule_12_lead_lag_v3
import src.strategy.rules.rule_12_lead_lag.v2 as rule_12_lead_lag_v2
import src.strategy.rules.rule_02_bollinger_band.v1 as rule_02_bollinger_band_v1
import src.strategy.rules.rule_03_ou_spread.v1 as rule_03_ou_spread_v1
import src.strategy.rules.rule_04_arima_forecast.v1 as rule_04_arima_forecast_v1
import src.strategy.rules.rule_05_fft_cycle.v1 as rule_05_fft_cycle_v1
import src.strategy.rules.rule_06_kalman_velocity.v1 as rule_06_kalman_velocity_v1
import src.strategy.rules.rule_07_order_book_imbalance.v1 as rule_07_order_book_imbalance_v1
import src.strategy.rules.rule_08_roc_momentum.v1 as rule_08_roc_momentum_v1
import src.strategy.rules.rule_09_markov_chain.v1 as rule_09_markov_chain_v1
import src.strategy.rules.rule_10_cnn_forecast.v1 as rule_10_cnn_forecast_v1
import src.strategy.rules.rule_11_dqn_agent.v1 as rule_11_dqn_agent_v1
import src.strategy.rules.rule_12_lead_lag.v1 as rule_12_lead_lag_v1
from src.agent.models import BuySignal, PairData, SellSignal

MarketData = dict[str, PairData]
Signal = BuySignal | SellSignal

ACTIVE_RULES: list[ModuleType] = [
    rule_01_spread_compression_v1,
    rule_02_bollinger_band_v1,
    rule_03_ou_spread_v1,
    rule_04_arima_forecast_v1,
    rule_05_fft_cycle_v1,
    rule_06_kalman_velocity_v1,
    rule_07_order_book_imbalance_v1,
    rule_08_roc_momentum_v1,
    rule_09_markov_chain_v1,
    rule_10_cnn_forecast_v1,
    rule_11_dqn_agent_v1,
    rule_12_lead_lag_v1,
    rule_12_lead_lag_v2,
    rule_12_lead_lag_v3,
]


def find_signals(data: MarketData) -> list[Signal]:
    signals: list[Signal] = []
    for rule in ACTIVE_RULES:
        try:
            signals.extend(rule.signal(data))
        except Exception:  # noqa: BLE001
            logging.exception("Rule %s raised an exception", rule.__name__)
    return signals
