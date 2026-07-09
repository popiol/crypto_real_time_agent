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

import src.strategy.rules.rule_05_fft_cycle.v2 as rule_05_fft_cycle_v2
import src.strategy.rules.rule_16_rsi_divergence_with_moving_ave.v1 as rule_16_rsi_divergence_with_moving_ave_v1
import src.strategy.rules.rule_15_adx_trend_strength_with_ema_cr.v1 as rule_15_adx_trend_strength_with_ema_cr_v1
import src.strategy.rules.rule_12_lead_lag.v5 as rule_12_lead_lag_v5
import src.strategy.rules.rule_12_lead_lag.v4 as rule_12_lead_lag_v4
import src.strategy.rules.rule_12_lead_lag.v3 as rule_12_lead_lag_v3
import src.strategy.rules.rule_12_lead_lag.v2 as rule_12_lead_lag_v2
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
    rule_12_lead_lag_v2,
    rule_12_lead_lag_v3,
    rule_12_lead_lag_v4,
    rule_05_fft_cycle_v2,
    rule_12_lead_lag_v5,
    rule_15_adx_trend_strength_with_ema_cr_v1,
    rule_16_rsi_divergence_with_moving_ave_v1,
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
