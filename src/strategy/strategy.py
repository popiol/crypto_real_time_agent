"""Strategy engine — signal detection.

find_signals() is the only public entry point. It calls signal(data) on the
single currently active rule module.

Exactly one rule is active at a time (ACTIVE_RULE), matching the Strategy
Updater's one-hypothesis-per-cycle learning loop: each cycle either fixes
the current rule (a new version replaces it) or replaces it outright with a
new rule concept. Previous rule files are kept on disk under
strategy/rules/ for signal traceability but are no longer executed once
replaced.
"""

from __future__ import annotations

import logging

import src.strategy.rules.HighVolatilityDipBuy.v2 as ACTIVE_RULE
from src.agent.models import BuySignal, MarketData, SellSignal

Signal = BuySignal | SellSignal


def find_signals(data: MarketData) -> list[Signal]:
    parts = ACTIVE_RULE.__name__.split(".")
    rule_id = f"{parts[-2]}_{parts[-1]}"
    signals: list[Signal] = []
    try:
        for signal in ACTIVE_RULE.signal(data):
            signal.rule_id = rule_id
            signals.append(signal)
    except Exception:  # noqa: BLE001
        logging.exception("Rule %s raised an exception", ACTIVE_RULE.__name__)
    return signals
