"""Strategy Updater pipeline orchestrator.

Runs all 8 steps sequentially. Each step reads its inputs from persisted
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
    step1_analyze_results,
    step2_analyze_rules,
    step3_compare_versions,
    step4_derive_conclusions,
    step5_update_plan,
    step6_generate_ideas,
    step7_evaluate_ideas,
    step8_implement_idea,
)

logger = logging.getLogger(__name__)

_STEPS: list[tuple[str, Callable]] = [
    ("1 analyze_results", step1_analyze_results.run),
    ("2 analyze_rules", step2_analyze_rules.run),
    ("3 compare_versions", step3_compare_versions.run),
    ("4 derive_conclusions", step4_derive_conclusions.run),
    ("5 update_plan", step5_update_plan.run),
    ("6 generate_ideas", step6_generate_ideas.run),
    ("7 evaluate_ideas", step7_evaluate_ideas.run),
    ("8 implement_idea", step8_implement_idea.run),
]


_STATE_FILES = [
    "signal_evaluation.json",
    "rule_descriptions.json",
    "rule_evaluation.json",
    "version_comparison.json",
    "conclusions.json",
    "long_term_plan.json",
    "idea_backlog.json",
]


def run(config: AppConfig) -> None:
    """Execute the full 8-step Strategy Updater pipeline."""
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
