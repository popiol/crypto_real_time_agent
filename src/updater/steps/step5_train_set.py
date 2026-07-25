"""Train set step — append one sample per evaluated signal to train_set.json.

For each signal that now has an evaluated outcome AND has indicator values
available (from indicator_values.json), appends a TrainSample. Uses signal_id
for deduplication so re-runs don't produce duplicate entries.

Writes / updates: data/state/train_set.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent import storage
from src.agent.models import AppConfig
from src.updater import paths
from src.updater.models import EpisodicTrace, TrainSample, TrainSet

logger = logging.getLogger(__name__)


def run(config: AppConfig, state_dir: Path) -> None:
    values_path = paths.indicator_values(state_dir)
    train_path = paths.train_set(state_dir)

    # Load indicator values (pair → {name → value})
    if not values_path.exists():
        logger.info("indicator_values.json not found; skipping train set update")
        return

    try:
        indicator_values: dict[str, dict[str, float | None]] = json.loads(
            values_path.read_text(encoding="utf-8")
        )
    except Exception:
        logger.warning("Could not read indicator_values.json", exc_info=True)
        return

    if not indicator_values:
        logger.info("No indicator values; skipping train set update")
        return

    # Load existing train set for deduplication
    train_set = _load_train_set(train_path)
    existing_signal_ids = {s.signal_id for s in train_set.samples}

    # Find evaluated signals with indicator values
    rule_cycle_map = _rule_cycle_map(state_dir)
    all_signals = storage.read_signals(config)
    new_samples: list[TrainSample] = []

    for signal in all_signals:
        outcome = signal.get("outcome")
        if outcome is None:
            continue  # not yet evaluated

        signal_id = signal.get("signal_id", "")
        if signal_id in existing_signal_ids:
            continue  # already in train set

        pair = signal.get("pair", "")
        if pair not in indicator_values:
            continue  # no indicator values for this pair

        gain_pct = outcome.get("gain_24h_pct") or outcome.get("gain_pct")
        if gain_pct is None:
            continue

        rule_id = signal.get("rule_id", "")
        new_samples.append(TrainSample(
            signal_id=signal_id,
            cycle_id=rule_cycle_map.get(rule_id, "unknown"),
            pair=pair,
            rule_id=rule_id,
            indicators=indicator_values[pair],
            target_gain_pct=gain_pct,
        ))

    if not new_samples:
        logger.info("No new signals to add to train set")
        return

    train_set.samples.extend(new_samples)
    train_path.write_text(train_set.model_dump_json(indent=2), encoding="utf-8")
    logger.info("train_set.json updated: +%d sample(s) (%d total)", len(new_samples), len(train_set.samples))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_train_set(path: Path) -> TrainSet:
    if not path.exists():
        return TrainSet()
    try:
        return TrainSet.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not parse train_set.json; starting fresh", exc_info=True)
        return TrainSet()


def _rule_cycle_map(state_dir: Path) -> dict[str, str]:
    """Return {rule_id: cycle_id} by scanning episodic traces.

    Each trace records the rule_id implemented in that cycle, so this lets
    every train sample carry the cycle_id of the cycle that actually
    produced its signal's rule — rather than whatever cycle happens to be
    the most recent one at the time this step runs.
    """
    mapping: dict[str, str] = {}
    traces_dir = paths.traces_dir(state_dir)
    if not traces_dir.exists():
        return mapping
    for path in traces_dir.glob("*.json"):
        try:
            trace = EpisodicTrace.model_validate_json(path.read_text(encoding="utf-8"))
            mapping[trace.rule_id] = trace.cycle_id
        except Exception:
            logger.warning("Could not parse trace %s", path, exc_info=True)
    return mapping
