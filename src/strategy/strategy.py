"""Strategy engine — signal detection.

find_signals() is the only public entry point. It resolves the currently
active rule from persisted state (data/state/plan.json's rule_id) and calls
signal(data) on it.

Exactly one rule is active at a time, matching the Strategy Updater's
one-hypothesis-per-cycle learning loop: each cycle either fixes the current
rule (a new version replaces it) or replaces it outright with a new rule
concept. The active rule is *state*, not code: the Strategy Updater never
edits this file — it only writes plan.json, and the rule module is imported
dynamically from there on every call. Previous rule files are kept on disk
under strategy/rules/ for signal traceability but are no longer executed
once replaced.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from pathlib import Path
from types import ModuleType

from src.agent.models import AppConfig, BuySignal, MarketData, SellSignal
from src.updater import paths

logger = logging.getLogger(__name__)

Signal = BuySignal | SellSignal


def rule_id_to_import_path(rule_id: str) -> str:
    """Convert 'rule_01_spread_compression_v1' -> 'rule_01_spread_compression.v1'."""
    m = re.match(r"^(.+)_(v\d+)$", rule_id)
    return f"{m.group(1)}.{m.group(2)}" if m else rule_id


def get_active_rule(config: AppConfig) -> ModuleType | None:
    """Dynamically import and return the currently active rule module.

    Returns None if no rule has been implemented yet (first-ever cycle) —
    plan.json's rule_id stays null until step6_implement_rule.py's step
    successfully implements one.
    """
    plan_path = paths.plan(Path(config.state_dir))
    if not plan_path.exists():
        return None
    try:
        rule_id = json.loads(plan_path.read_text(encoding="utf-8"))["rule_id"]
    except Exception:
        logger.warning("Could not read plan.json for active rule", exc_info=True)
        return None
    if rule_id is None:
        return None
    import_path = rule_id_to_import_path(rule_id)
    try:
        return importlib.import_module(f"src.strategy.rules.{import_path}")
    except Exception:
        logger.exception("Could not import active rule '%s'", rule_id)
        return None


def find_signals(data: MarketData, config: AppConfig) -> list[Signal]:
    rule = get_active_rule(config)
    if rule is None:
        return []
    parts = rule.__name__.split(".")
    rule_id = f"{parts[-2]}_{parts[-1]}"
    signals: list[Signal] = []
    try:
        for signal in rule.signal(data):
            signal.rule_id = rule_id
            signals.append(signal)
    except Exception:  # noqa: BLE001
        logger.exception("Rule %s raised an exception", rule.__name__)
    return signals
