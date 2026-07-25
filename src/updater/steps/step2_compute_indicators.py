"""Compute indicators step — execute indicator functions against tier data.

For each pair that emitted a signal in the previous cycle, runs all indicator
functions from indicator_set.json against the pair's current warm/cold tier
data. Execution errors per indicator are caught and logged; they do not halt
the pipeline.

Writes: data/state/indicator_values.json
    {pair: {indicator_name: float | null}}
"""

from __future__ import annotations

import json
import logging
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.agent import storage
from src.agent.models import AppConfig, PairData
from src.updater import paths
from src.updater.models import IndicatorSet

logger = logging.getLogger(__name__)


def run(config: AppConfig, state_dir: Path) -> None:
    values_path = paths.indicator_values(state_dir)
    set_path = paths.indicator_set(state_dir)
    if not set_path.exists():
        logger.info("indicator_set.json not found; skipping indicator computation")
        values_path.write_text("{}", encoding="utf-8")
        return

    indicator_set = IndicatorSet.model_validate_json(set_path.read_text(encoding="utf-8"))
    if not indicator_set.indicators:
        logger.info("Indicator set is empty; skipping computation")
        values_path.write_text("{}", encoding="utf-8")
        return

    # Identify pairs that have signals with unresolved outcomes (emitted in the past 24h)
    all_signals = storage.read_signals(config)
    pairs = list({s["pair"] for s in all_signals if s.get("outcome") is None})

    if not pairs:
        logger.info("No recent signals found; skipping indicator computation")
        values_path.write_text("{}", encoding="utf-8")
        return

    results: dict[str, dict[str, float | None]] = {}
    for pair in pairs:
        pair_data = PairData(
            hot=storage.read_ticks(pair, config),
            warm=storage.read_warm_candles(pair, config),
            cold=storage.read_cold_months(pair, config),
        )
        results[pair] = _compute_for_pair(indicator_set, pair_data)

    dead_names = _all_null_indicator_names(indicator_set, results)
    if dead_names:
        indicator_set = _drop_indicators(indicator_set, dead_names, set_path)
        for pair_values in results.values():
            for name in dead_names:
                pair_values.pop(name, None)

    values_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(
        "indicator_values.json written: %d pair(s), %d indicator(s) each",
        len(results),
        len(indicator_set.indicators),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _all_null_indicator_names(
    indicator_set: IndicatorSet, results: dict[str, dict[str, float | None]]
) -> set[str]:
    """Return indicator names that produced None for every pair this cycle."""
    if not results:
        return set()
    return {
        indicator.name
        for indicator in indicator_set.indicators
        if all(pair_values.get(indicator.name) is None for pair_values in results.values())
    }


def _drop_indicators(
    indicator_set: IndicatorSet, dead_names: set[str], set_path: Path
) -> IndicatorSet:
    """Remove all-null indicators from the persisted indicator set."""
    updated = indicator_set.model_copy(
        update={
            "indicators": [
                i for i in indicator_set.indicators if i.name not in dead_names
            ],
            "version": str(uuid.uuid4()),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    set_path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "Removed %d all-null indicator(s) from indicator_set.json: %s",
        len(dead_names),
        sorted(dead_names),
    )
    return updated


def _compute_for_pair(
    indicator_set: IndicatorSet, pair_data: PairData
) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for indicator in indicator_set.indicators:
        values[indicator.name] = _run_indicator(indicator.name, indicator.code, pair_data)
    return values


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
