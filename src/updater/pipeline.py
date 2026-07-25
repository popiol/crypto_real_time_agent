"""Strategy Updater pipeline orchestrator.

Runs all active steps sequentially. Each step reads its inputs from persisted
state files and writes its output before the next step begins, making the
pipeline resumable and auditable.

Requires one of:
  pip install langchain-core langchain-google-genai   # for gemini-* models
  pip install langchain-core langchain-anthropic      # for claude-* models
  pip install langchain-core langchain-openai         # for gpt-* / o* models
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.agent.models import AppConfig
from src.updater.steps import (
    step1_indicator_set,
    step2_compute_indicators,
    step3_analyze_results,
    step4_analyze_rules,
    step5_train_set,
    step6_episodic_trace,
    step7_implement_idea,
)

logger = logging.getLogger(__name__)

_STEPS: list[tuple[str, Callable]] = [
    ("1 indicator_set", step1_indicator_set.run),
    ("2 compute_indicators", step2_compute_indicators.run),
    ("3 analyze_results", step3_analyze_results.run),
    ("4 analyze_rules", step4_analyze_rules.run),
    ("5 train_set", step5_train_set.run),
    ("6 episodic_trace", step6_episodic_trace.run),
    ("7 implement_idea", step7_implement_idea.run),
]


_STATE_FILES = [
    "signal_evaluation.json",
    "rule_descriptions.json",
    "rule_evaluation.json",
    "indicator_set.json",
    "indicator_values.json",
    "train_set.json",
    "last_implemented.json",
]


def run(config: AppConfig) -> None:
    """Execute the Strategy Updater pipeline."""
    state_dir = Path(config.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Strategy Updater pipeline starting")
    for name, step_fn in _STEPS:
        logger.info("Step %s", name)
        try:
            step_fn(config, state_dir)
        except Exception:
            logger.exception("Step %s failed; continuing with remaining steps", name)
    logger.info("Strategy Updater pipeline complete")

    _archive_state(state_dir)


def _archive_state(state_dir: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    history_dir = state_dir / "history" / ts
    archived = 0
    for name in _STATE_FILES:
        src = state_dir / name
        if src.exists():
            history_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, history_dir / name)
            archived += 1
    if archived:
        logger.info("Archived %d state file(s) to %s", archived, history_dir)
