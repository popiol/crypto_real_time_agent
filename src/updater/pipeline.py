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

from src.agent import storage
from src.agent.models import AppConfig
from src.updater import paths
from src.updater.steps import (
    step1_indicator_set,
    step2_evaluate_rule,
    step3_train_set,
    step4_relation_analysis,
    step5_generate_idea,
    step6_implement_rule,
)

logger = logging.getLogger(__name__)

_STEPS: list[tuple[str, Callable]] = [
    ("1 indicator_set", step1_indicator_set.run),
    ("2 evaluate_rule", step2_evaluate_rule.run),
    ("3 train_set", step3_train_set.run),
    ("4 relation_analysis", step4_relation_analysis.run),
    ("5 generate_idea", step5_generate_idea.run),
    ("6 implement_rule", step6_implement_rule.run),
]


_STATE_FILES = [
    paths.RULE_DESCRIPTIONS,
    paths.RULE_EVALUATION,
    paths.INDICATOR_SET,
    paths.TRAIN_SET,
    paths.PLAN,
]


def run(config: AppConfig) -> None:
    """Execute the Strategy Updater pipeline."""
    state_dir = Path(config.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    # Computed once and threaded through every step, rather than each step
    # independently re-deriving "what cycle is this" from quote data.
    latest_quote = storage.latest_quote_time(config)
    if latest_quote is None:
        logger.info("No quote data yet; skipping Strategy Updater pipeline run")
        return
    cycle_id = latest_quote.strftime("%Y-%m-%dT%H-%M-%S")

    logger.info("Strategy Updater pipeline starting (cycle_id=%s)", cycle_id)
    for name, step_fn in _STEPS:
        logger.info("Step %s", name)
        try:
            step_fn(config, state_dir, cycle_id)
        except Exception:
            logger.exception("Step %s failed; continuing with remaining steps", name)
    logger.info("Strategy Updater pipeline complete")

    _archive_state(state_dir)


def _archive_state(state_dir: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    history_dir = paths.history_dir(state_dir, ts)
    archived = 0
    for name in _STATE_FILES:
        src = state_dir / name
        if src.exists():
            history_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, history_dir / name)
            archived += 1
    if archived:
        logger.info("Archived %d state file(s) to %s", archived, history_dir)
