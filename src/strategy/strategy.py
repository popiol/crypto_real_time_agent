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

import src.strategy.rules.rule_36_bollinger_band_rejection_with_.v1 as rule_36_bollinger_band_rejection_with__v1
import src.strategy.rules.rule_37_bollinger_band_breach_with_eng.v1 as rule_37_bollinger_band_breach_with_eng_v1
import src.strategy.rules.rule_35_bollinger_band_rejection_with_.v1 as rule_35_bollinger_band_rejection_with__v1
import src.strategy.rules.rule_34_bollinger_band_rejection_with_.v1 as rule_34_bollinger_band_rejection_with__v1
import src.strategy.rules.rule_33_bollinger_band_rejection_with_.v1 as rule_33_bollinger_band_rejection_with__v1
import src.strategy.rules.rule_24_bollinger_band_breach_with_mfi.v2 as rule_24_bollinger_band_breach_with_mfi_v2
import src.strategy.rules.rule_28_bollinger_band_reversal_with_c.v1 as rule_28_bollinger_band_reversal_with_c_v1
import src.strategy.rules.rule_26_bollinger_band_breach_with_mfi.v1 as rule_26_bollinger_band_breach_with_mfi_v1
import src.strategy.rules.rule_02_bollinger_band.v2 as rule_02_bollinger_band_v2
from src.agent.models import BuySignal, MarketData, SellSignal

Signal = BuySignal | SellSignal

ACTIVE_RULES: list[ModuleType] = [
    rule_02_bollinger_band_v2,
    rule_26_bollinger_band_breach_with_mfi_v1,
    rule_28_bollinger_band_reversal_with_c_v1,
    rule_24_bollinger_band_breach_with_mfi_v2,
    rule_33_bollinger_band_rejection_with__v1,
    rule_34_bollinger_band_rejection_with__v1,
    rule_35_bollinger_band_rejection_with__v1,
    rule_36_bollinger_band_rejection_with__v1,
    rule_37_bollinger_band_breach_with_eng_v1,
]


def find_signals(data: MarketData) -> list[Signal]:
    signals: list[Signal] = []
    for rule in ACTIVE_RULES:
        parts = rule.__name__.split(".")
        rule_id = f"{parts[-2]}_{parts[-1]}"
        try:
            for signal in rule.signal(data):
                signal.rule_id = rule_id
                signals.append(signal)
        except Exception:  # noqa: BLE001
            logging.exception("Rule %s raised an exception", rule.__name__)
    return signals
