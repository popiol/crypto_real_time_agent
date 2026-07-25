"""Train set step — append one sample per evaluated signal to train_set.json.

For each signal that now has an evaluated outcome, appends a TrainSample
using the indicator values already captured on the signal at emission time
(src.agent.loop._attach_indicators) — not recomputed here, since by the time
a signal resolves (up to 20 days later) the tier data that produced its
original indicator readings has long since rolled off. Uses signal_id for
deduplication so re-runs don't produce duplicate entries.

Also prunes indicators that come back null across every new sample this
cycle from indicator_set.json (moved here from the old standalone compute-
indicators step, since that's the only place a batch of fresh readings is
available now that indicators are computed one signal at a time).

Writes / updates: data/state/train_set.json, data/state/indicator_set.json (indicators pruned)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.agent import storage
from src.agent.models import AppConfig
from src.updater import paths
from src.updater.models import IndicatorSet, TrainSample, TrainSet

logger = logging.getLogger(__name__)


def run(config: AppConfig, state_dir: Path, cycle_id: str) -> None:
    train_path = paths.train_set(state_dir)

    # Load existing train set for deduplication
    train_set = _load_train_set(train_path)
    existing_signal_ids = {s.signal_id for s in train_set.samples}

    all_signals = storage.read_signals(config)
    new_samples: list[TrainSample] = []

    for signal in all_signals:
        outcome = signal.get("outcome")
        if outcome is None:
            continue  # not yet evaluated

        signal_id = signal.get("signal_id", "")
        if signal_id in existing_signal_ids:
            continue  # already in train set

        indicator_values = signal.get("indicators") or {}
        if not indicator_values:
            continue  # signal predates indicator capture at emission time

        gain_pct = outcome.get("gain_24h_pct") or outcome.get("gain_pct")
        if gain_pct is None:
            continue

        new_samples.append(TrainSample(
            signal_id=signal_id,
            cycle_id=cycle_id,
            pair=signal.get("pair", ""),
            rule_id=signal.get("rule_id", ""),
            indicators=indicator_values,
            target_gain_pct=gain_pct,
        ))

    if not new_samples:
        logger.info("No new signals to add to train set")
        return

    train_set.samples.extend(new_samples)
    train_path.write_text(train_set.model_dump_json(indent=2), encoding="utf-8")
    logger.info("train_set.json updated: +%d sample(s) (%d total)", len(new_samples), len(train_set.samples))

    _prune_dead_indicators(state_dir, new_samples)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_train_set(path: Path) -> TrainSet:
    if not path.exists():
        return TrainSet()
    try:
        return TrainSet.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not parse train_set.json; starting fresh", exc_info=True)
        return TrainSet()


def _prune_dead_indicators(state_dir: Path, new_samples: list[TrainSample]) -> None:
    """Remove indicators that were null across every new sample this cycle."""
    set_path = paths.indicator_set(state_dir)
    if not set_path.exists():
        return
    try:
        indicator_set = IndicatorSet.model_validate_json(set_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read indicator_set.json", exc_info=True)
        return
    if not indicator_set.indicators:
        return

    dead_names = {
        indicator.name
        for indicator in indicator_set.indicators
        if all(s.indicators.get(indicator.name) is None for s in new_samples)
    }
    if not dead_names:
        return

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
