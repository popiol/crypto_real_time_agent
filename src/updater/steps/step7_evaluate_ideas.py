"""Step 7 — Evaluate ideas.

In a single LLM call, re-scores every idea in the backlog (proposed and
previously evaluated) against the current long-term plan.

Writes idea_backlog.json.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.agent.models import AppConfig
from src.updater.llm import llm_structured
from src.updater.models import IdeaBacklog, LongTermPlan

logger = logging.getLogger(__name__)

_REJECT_THRESHOLD = 0.2


def run(config: AppConfig, state_dir: Path) -> None:
    backlog_path = state_dir / "idea_backlog.json"
    if not backlog_path.exists():
        logger.info("idea_backlog.json not found; skipping step 7")
        return

    backlog = IdeaBacklog.model_validate_json(backlog_path.read_text(encoding="utf-8"))
    scoreable = [i for i in backlog.ideas if i.status in ("proposed", "evaluated")]
    if not scoreable:
        logger.info("No ideas to score; skipping step 7")
        return

    plan_text = "No plan available."
    plan_path = state_dir / "long_term_plan.json"
    if plan_path.exists():
        try:
            plan = LongTermPlan.model_validate_json(
                plan_path.read_text(encoding="utf-8")
            )
            plan_text = plan.model_dump_json(indent=2)
        except Exception:
            pass

    result = llm_structured(
        model=config.llm_model,
        system=(
            "You are a quantitative trading researcher scoring rule ideas "
            "for an autonomous crypto trading agent."
        ),
        user=(
            "Score every idea in the backlog against the long-term plan. "
            "For each idea assign a score 0.0–1.0 (potential value × feasibility). "
            f"Mark ideas with score < {_REJECT_THRESHOLD} as 'rejected', "
            "all others as 'evaluated'. "
            "Return the complete updated idea list.\n\n"
            f"Long-term plan:\n{plan_text}\n\n"
            f"Ideas to score:\n{IdeaBacklog(ideas=scoreable).model_dump_json(indent=2)}"
        ),
        output_type=IdeaBacklog,
    )

    # Preserve implemented/rejected status from existing backlog; only update scoreable ones
    scored_by_id = {i.idea_id: i for i in result.ideas}
    for idea in backlog.ideas:
        if idea.status == "implemented" or idea.idea_id not in scored_by_id:
            scored_by_id[idea.idea_id] = idea

    final_backlog = IdeaBacklog(ideas=list(scored_by_id.values()))
    backlog_path.write_text(final_backlog.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "idea_backlog.json written (%d ideas, %d scored this run)",
        len(final_backlog.ideas),
        len(scoreable),
    )
