"""Step 6 — Generate ideas.

Generates new rule ideas until stop conditions are met:
  - At least 3 open ideas AND highest-scored open idea has score >= 0.6, OR
  - 3 new ideas generated this cycle.

Writes idea_backlog.json.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from src.agent.models import AppConfig
from src.strategy.strategy import ACTIVE_RULES
from src.updater.llm import llm_structured
from src.updater.models import IdeaBacklog, LongTermPlan, RuleDescriptions, RuleIdea

logger = logging.getLogger(__name__)

_MAX_NEW_IDEAS = 3
_MIN_OPEN_IDEAS = 3
_MIN_TOP_SCORE = 0.6


def run(config: AppConfig, state_dir: Path) -> None:
    plan_path = state_dir / "long_term_plan.json"
    if not plan_path.exists():
        logger.info("long_term_plan.json not found; skipping step 6")
        return

    plan = LongTermPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    backlog = _load_backlog(state_dir / "idea_backlog.json")

    if _stop_conditions_met(backlog.ideas):
        logger.info("Step 6 stop conditions already met; skipping idea generation")
        return

    rule_summary = _rule_summary(state_dir)
    open_ideas_summary = _open_ideas_summary(backlog.ideas)

    generated = 0
    while generated < _MAX_NEW_IDEAS:
        if _stop_conditions_met(backlog.ideas):
            break
        try:
            idea = _generate_one(plan, rule_summary, open_ideas_summary, config.llm_model)
            backlog.ideas.append(idea)
            open_ideas_summary = _open_ideas_summary(backlog.ideas)
            generated += 1
            logger.info("Generated idea: %s (%s)", idea.title, idea.idea_id)
        except Exception:
            logger.warning("Idea generation LLM call failed", exc_info=True)
            break

    (state_dir / "idea_backlog.json").write_text(
        backlog.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("idea_backlog.json written (%d total ideas, %d new)", len(backlog.ideas), generated)


def _stop_conditions_met(ideas: list[RuleIdea]) -> bool:
    open_ideas = [i for i in ideas if i.status in ("proposed", "evaluated")]
    if len(open_ideas) < _MIN_OPEN_IDEAS:
        return False
    scored = [i.score for i in open_ideas if i.score is not None]
    return bool(scored) and max(scored) >= _MIN_TOP_SCORE


def _generate_one(
    plan: LongTermPlan,
    rule_summary: str,
    open_ideas_summary: str,
    model: str,
) -> RuleIdea:
    new_id = str(uuid.uuid4())
    result = llm_structured(
        model=model,
        system=(
            "You are a quantitative trading researcher designing buy-signal rules "
            "for a crypto trading agent."
        ),
        user=(
            "Generate one new trading rule idea that aligns with the long-term plan "
            "and has not already been proposed.\n\n"
            f"Long-term plan:\n{plan.model_dump_json(indent=2)}\n\n"
            f"Currently implemented rules:\n{rule_summary}\n\n"
            f"Open ideas already in backlog:\n{open_ideas_summary}\n\n"
            f"Use this idea_id: {new_id}\n"
            "Set status to 'proposed' and score to null.\n"
            "For a new_rule idea, set target_rule to null. "
            "For a modify_rule idea, set target_rule to the rule_id to modify."
        ),
        output_type=RuleIdea,
    )
    result.idea_id = new_id
    result.status = "proposed"
    result.score = None
    return result


def _load_backlog(path: Path) -> IdeaBacklog:
    if path.exists():
        try:
            return IdeaBacklog.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not parse idea_backlog.json; starting fresh")
    return IdeaBacklog(ideas=[])


def _rule_summary(state_dir: Path) -> str:
    desc_path = state_dir / "rule_descriptions.json"
    if desc_path.exists():
        try:
            descs = RuleDescriptions.model_validate_json(desc_path.read_text(encoding="utf-8"))
            return json.dumps([r.model_dump() for r in descs.rules], indent=2)
        except Exception:
            pass
    return ", ".join(fn.__module__.split(".")[-1] for fn in ACTIVE_RULES)


def _open_ideas_summary(ideas: list[RuleIdea]) -> str:
    open_ideas = [i for i in ideas if i.status in ("proposed", "evaluated")]
    if not open_ideas:
        return "None"
    return json.dumps([{"title": i.title, "description": i.description} for i in open_ideas], indent=2)
