"""Indicator execution — shared by signal creation (src/agent/loop.py) and
the Strategy Updater's train-set step (design.md §8.2 Step 7).

Indicators are LLM-generated Python snippets (design.md §8.2 Step 1) stored
in indicator_set.json. This module is the one place that actually executes
that generated code against real PairData.
"""

from __future__ import annotations

import logging
import types
from pathlib import Path

from src.agent.models import PairData
from src.updater import paths
from src.updater.models import IndicatorSet

logger = logging.getLogger(__name__)


def load_indicator_set(state_dir: Path) -> IndicatorSet | None:
    set_path = paths.indicator_set(state_dir)
    if not set_path.exists():
        return None
    try:
        return IndicatorSet.model_validate_json(set_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read indicator_set.json", exc_info=True)
        return None


def compute_indicators(
    indicator_set: IndicatorSet, pair_data: PairData
) -> dict[str, float | None]:
    return {
        indicator.name: _run_indicator(indicator.name, indicator.code, pair_data)
        for indicator in indicator_set.indicators
    }


def _run_indicator(name: str, code: str, pair_data: PairData) -> float | None:
    try:
        module = types.ModuleType(f"_indicator_{name}")
        exec(compile(code, f"<indicator:{name}>", "exec"), module.__dict__)  # noqa: S102
        compute_fn = getattr(module, "compute", None)
        if compute_fn is None:
            logger.warning("Indicator '%s' has no `compute` function; skipping", name)
            return None
        result = compute_fn(pair_data)
        if result is None:
            return None
        if not isinstance(result, (int, float)):
            logger.warning(
                "Indicator '%s' returned non-numeric value %r; treating as None",
                name,
                result,
            )
            return None
        return float(result)
    except Exception:
        logger.warning("Indicator '%s' raised an exception", name, exc_info=True)
        return None
