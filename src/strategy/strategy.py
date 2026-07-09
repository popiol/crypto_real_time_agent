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

import src.strategy.rules.rule_04_arima_forecast.v2 as rule_04_arima_forecast_v2
import src.strategy.rules.rule_24_adx_filtered_moving_average_cr.v1 as rule_24_adx_filtered_moving_average_cr_v1
import src.strategy.rules.rule_22_triple_moving_average_crossove.v2 as rule_22_triple_moving_average_crossove_v2
import src.strategy.rules.rule_21_keltner_channel_breakout_with_.v2 as rule_21_keltner_channel_breakout_with__v2
import src.strategy.rules.rule_23_ichimoku_cloud_breakout_with_r.v1 as rule_23_ichimoku_cloud_breakout_with_r_v1
import src.strategy.rules.rule_02_bollinger_band.v1 as rule_02_bollinger_band_v1
import src.strategy.rules.rule_04_arima_forecast.v1 as rule_04_arima_forecast_v1
import src.strategy.rules.rule_05_fft_cycle.v1 as rule_05_fft_cycle_v1
import src.strategy.rules.rule_09_markov_chain.v1 as rule_09_markov_chain_v1
import src.strategy.rules.rule_12_lead_lag.v1 as rule_12_lead_lag_v1
from src.agent.models import MarketData, BuySignal, SellSignal

Signal = BuySignal | SellSignal

ACTIVE_RULES: list[ModuleType] = [
    rule_02_bollinger_band_v1,
    rule_04_arima_forecast_v1,
    rule_05_fft_cycle_v1,
    rule_09_markov_chain_v1,
    rule_12_lead_lag_v1,
    rule_04_arima_forecast_v2,
    rule_23_ichimoku_cloud_breakout_with_r_v1,
    rule_21_keltner_channel_breakout_with__v2,
    rule_22_triple_moving_average_crossove_v2,
    rule_24_adx_filtered_moving_average_cr_v1,
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
