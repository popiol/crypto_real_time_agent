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

import src.strategy.rules.rule_02_bollinger_band.v1 as rule_02_bollinger_band_v1
import src.strategy.rules.rule_02_bollinger_band.v8 as rule_02_bollinger_band_v8
import src.strategy.rules.rule_19_bollinger_band_with_volume_spi.v3 as rule_19_bollinger_band_with_volume_spi_v3
import src.strategy.rules.rule_22_bollinger_band_re_entry_with_v.v1 as rule_22_bollinger_band_re_entry_with_v_v1
import src.strategy.rules.rule_21_bollinger_band_breakout_with_a.v1 as rule_21_bollinger_band_breakout_with_a_v1
import src.strategy.rules.rule_15_vwap_bands_trading_rule.v3 as rule_15_vwap_bands_trading_rule_v3
import src.strategy.rules.rule_02_bollinger_band.v7 as rule_02_bollinger_band_v7
import src.strategy.rules.rule_19_bollinger_band_with_volume_spi.v2 as rule_19_bollinger_band_with_volume_spi_v2
import src.strategy.rules.rule_18_average_directional_index_adx_.v2 as rule_18_average_directional_index_adx__v2
import src.strategy.rules.rule_02_bollinger_band.v6 as rule_02_bollinger_band_v6
import src.strategy.rules.rule_20_keltner_channel_mean_reversion.v2 as rule_20_keltner_channel_mean_reversion_v2
import src.strategy.rules.rule_20_keltner_channel_mean_reversion.v1 as rule_20_keltner_channel_mean_reversion_v1
import src.strategy.rules.rule_02_bollinger_band.v5 as rule_02_bollinger_band_v5
import src.strategy.rules.rule_19_bollinger_band_with_volume_spi.v1 as rule_19_bollinger_band_with_volume_spi_v1
import src.strategy.rules.rule_02_bollinger_band.v4 as rule_02_bollinger_band_v4
import src.strategy.rules.rule_17_money_flow_index_mfi_overbough.v2 as rule_17_money_flow_index_mfi_overbough_v2
import src.strategy.rules.rule_02_bollinger_band.v3 as rule_02_bollinger_band_v3
import src.strategy.rules.rule_18_average_directional_index_adx_.v1 as rule_18_average_directional_index_adx__v1
import src.strategy.rules.rule_17_money_flow_index_mfi_overbough.v1 as rule_17_money_flow_index_mfi_overbough_v1
import src.strategy.rules.rule_15_vwap_bands_trading_rule.v2 as rule_15_vwap_bands_trading_rule_v2
import src.strategy.rules.rule_15_vwap_bands_trading_rule.v1 as rule_15_vwap_bands_trading_rule_v1
import src.strategy.rules.rule_02_bollinger_band.v2 as rule_02_bollinger_band_v2
import src.strategy.rules.rule_04_arima_forecast.v1 as rule_04_arima_forecast_v1
from src.agent.models import BuySignal, MarketData, SellSignal

Signal = BuySignal | SellSignal

ACTIVE_RULES: list[ModuleType] = [
    rule_02_bollinger_band_v1,
    rule_04_arima_forecast_v1,
    rule_02_bollinger_band_v2,
    rule_15_vwap_bands_trading_rule_v1,
    rule_15_vwap_bands_trading_rule_v2,
    rule_17_money_flow_index_mfi_overbough_v1,
    rule_18_average_directional_index_adx__v1,
    rule_02_bollinger_band_v3,
    rule_17_money_flow_index_mfi_overbough_v2,
    rule_02_bollinger_band_v4,
    rule_19_bollinger_band_with_volume_spi_v1,
    rule_02_bollinger_band_v5,
    rule_20_keltner_channel_mean_reversion_v1,
    rule_20_keltner_channel_mean_reversion_v2,
    rule_02_bollinger_band_v6,
    rule_18_average_directional_index_adx__v2,
    rule_19_bollinger_band_with_volume_spi_v2,
    rule_02_bollinger_band_v7,
    rule_15_vwap_bands_trading_rule_v3,
    rule_21_bollinger_band_breakout_with_a_v1,
    rule_22_bollinger_band_re_entry_with_v_v1,
    rule_19_bollinger_band_with_volume_spi_v3,
    rule_02_bollinger_band_v8,
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
