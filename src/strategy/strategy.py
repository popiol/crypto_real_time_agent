"""Strategy engine — buy signal detection.

find_buy_signals() is the only public entry point. It calls every registered
rule function and merges the results.

To add a rule: create src/strategy/rules/rule_NN_<name>.py, then import its
function below and append it to ACTIVE_RULES.
"""

from __future__ import annotations

import logging

from src.agent.models import BuySignal, PairData
from src.strategy.rules.rule_01_spread_compression import spread_compression_spike
from src.strategy.rules.rule_02_bollinger_band import bollinger_band_lower_touch
from src.strategy.rules.rule_03_ou_spread import ou_spread_compression
from src.strategy.rules.rule_04_arima_forecast import arima_price_forecast
from src.strategy.rules.rule_05_fft_cycle import fft_cycle_trough
from src.strategy.rules.rule_06_kalman_velocity import kalman_velocity_reversal

# ---------------------------------------------------------------------------
# Type alias for the data passed to each rule
# ---------------------------------------------------------------------------

# MarketData: maps pair name → PairData (hot + warm tiers)
MarketData = dict[str, PairData]


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

ACTIVE_RULES = [
    spread_compression_spike,
    bollinger_band_lower_touch,
    ou_spread_compression,
    arima_price_forecast,
    fft_cycle_trough,
    kalman_velocity_reversal,
]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def find_buy_signals(data: MarketData) -> list[BuySignal]:
    """Run all active rules against the current market data and return signals."""
    signals: list[BuySignal] = []
    for rule_fn in ACTIVE_RULES:
        try:
            signals.extend(rule_fn(data))
        except Exception:  # noqa: BLE001
            logging.exception("Rule %s raised an exception", rule_fn.__name__)
    return signals
