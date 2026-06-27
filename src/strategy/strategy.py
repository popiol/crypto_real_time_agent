"""Strategy engine — buy signal detection.

find_buy_signals() is the only public entry point. It calls every registered
rule function and merges the results.

Rules are registered by adding their function to ACTIVE_RULES below.
Deprecated rules are kept in this file but removed from ACTIVE_RULES.
"""

from __future__ import annotations

import logging

from src.agent.models import BuySignal, Tick

# ---------------------------------------------------------------------------
# Type alias for the data passed to each rule
# ---------------------------------------------------------------------------

# MarketData: dict mapping pair name → list of hot-tier ticks (most recent last)
MarketData = dict[str, list[Tick]]


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

ACTIVE_RULES: list = []  # populated below after rule definitions


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

# (no rules yet — the Strategy Updater will add them here)


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
