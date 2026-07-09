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

import src.strategy.rules.rule_02_bollinger_band.v3 as rule_02_bollinger_band_v3
import src.strategy.rules.rule_13_vwap_deviation_with_volume_con.v2 as rule_13_vwap_deviation_with_volume_con_v2
import src.strategy.rules.rule_08_roc_momentum.v3 as rule_08_roc_momentum_v3
import src.strategy.rules.rule_14_rsi_mid_range_crossover_for_fr.v1 as rule_14_rsi_mid_range_crossover_for_fr_v1
import src.strategy.rules.rule_08_roc_momentum.v2 as rule_08_roc_momentum_v2
import src.strategy.rules.rule_02_bollinger_band.v2 as rule_02_bollinger_band_v2
import src.strategy.rules.rule_07_order_book_imbalance.v2 as rule_07_order_book_imbalance_v2
from src.agent.models import BuySignal, PairData, SellSignal

MarketData = dict[str, PairData]
Signal = BuySignal | SellSignal

ACTIVE_RULES: list[ModuleType] = [
    rule_07_order_book_imbalance_v2,
    rule_02_bollinger_band_v2,
    rule_08_roc_momentum_v2,
    rule_14_rsi_mid_range_crossover_for_fr_v1,
    rule_02_bollinger_band_v3,
    rule_08_roc_momentum_v3,
    rule_13_vwap_deviation_with_volume_con_v2,
]


def find_signals(data: MarketData) -> list[Signal]:
    signals: list[Signal] = []
    for rule in ACTIVE_RULES:
        try:
            signals.extend(rule.signal(data))
        except Exception:  # noqa: BLE001
            logging.exception("Rule %s raised an exception", rule.__name__)
    return signals
