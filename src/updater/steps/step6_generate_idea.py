"""Generate rule idea — design.md §8.2 Step 6.

Given the relation analysis findings and the current cycle plan, generates
exactly one RuleIdea. For a 'fix' plan, target_rule is always the current
active rule (taken from plan.json) — never left to the LLM to guess, since
exactly one rule is ever active (see strategy.py).

run() is this pipeline.py stage's entry point: it reads relation_analysis.json,
skipping if there's nothing there for the current cycle_id (either step5's
run decided action=continue and produced nothing, or a stale file from a run
where nothing consumed it — either way, not fresh), otherwise generates one
idea and persists it to rule_idea.json for step7_implement_rule.py's step to
pick up.

Reads:
  data/state/relation_analysis.json
Writes:
  data/state/plan.json (only on idea-generation failure),
  data/state/rule_idea.json, data/state/relation_analysis.json (cleared)
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from src.agent.models import AppConfig
from src.updater import paths
from src.updater import plan_next_cycle as plan_next_cycle_step
from src.updater.llm import llm_structured
from src.updater.models import (
    Plan,
    PendingRelationAnalysis,
    PendingRuleIdea,
    RelationAnalysis,
    RuleIdea,
)
from src.updater.steps.step7_implement_rule import DATA_WINDOW_CONSTRAINT

logger = logging.getLogger(__name__)

_IDEA_GENERATION_SYSTEM = (
    "You are a quantitative trading strategist. "
    "Based on the relation analysis findings and the current cycle plan, "
    "generate exactly ONE concrete trading rule idea. "
    "The idea must be directly grounded in the observed indicator patterns. "
    "For 'fix', target the named rule and describe specific improvements. "
    "For 'new_rule', propose a genuinely different approach. "
    "Be precise: include thresholds, conditions, and expected market behaviour. "
    + DATA_WINDOW_CONSTRAINT
)


def run(config: AppConfig, state_dir: Path, cycle_id: str) -> None:
    pending = _load_pending_analysis(state_dir, cycle_id)
    if pending is None:
        logger.info("No fresh relation analysis for this cycle; skipping idea generation")
        return
    plan, analysis = pending.plan, pending.analysis

    try:
        idea = _generate_idea(analysis, plan, config)
        if not idea.idea_id:
            idea = idea.model_copy(update={"idea_id": str(uuid.uuid4())})
        logger.info(
            "Generated idea '%s' (kind=%s, target_rule=%s): %s",
            idea.title,
            idea.kind,
            idea.target_rule,
            idea.rationale,
        )
    except Exception:
        logger.exception("Idea generation failed; nothing for implement_rule to pick up")
        plan_next_cycle_step.write_next_cycle_plan(state_dir, plan, analysis, config)
        paths.relation_analysis(state_dir).unlink(missing_ok=True)
        return

    pending_idea = PendingRuleIdea(cycle_id=cycle_id, idea=idea, plan=plan, analysis=analysis)
    paths.rule_idea(state_dir).write_text(
        pending_idea.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("rule_idea.json written: idea='%s' (cycle_id=%s)", idea.title, cycle_id)
    paths.relation_analysis(state_dir).unlink(missing_ok=True)


def _load_pending_analysis(state_dir: Path, cycle_id: str) -> PendingRelationAnalysis | None:
    path = paths.relation_analysis(state_dir)
    if not path.exists():
        return None
    try:
        pending = PendingRelationAnalysis.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read relation_analysis.json", exc_info=True)
        return None
    if pending.cycle_id != cycle_id:
        logger.info(
            "relation_analysis.json is from cycle %s, not the current cycle %s; skipping",
            pending.cycle_id,
            cycle_id,
        )
        return None
    return pending


def _generate_idea(
    analysis: RelationAnalysis,
    plan: Plan,
    config: AppConfig,
) -> RuleIdea:
    plan_section = f"Current cycle plan:\nAction: {plan.action}\n"
    if plan.description:
        plan_section += f"Description: {plan.description}\n"

    user = (
        f"{plan_section}\n"
        f"Relation analysis findings:\n{analysis.model_dump_json(indent=2)}\n\n"
        "Generate exactly ONE rule idea based on these findings. "
        "If the plan action is 'fix', set kind='modify_rule' (target_rule is filled in "
        "automatically — do not invent one). Otherwise, kind='new_rule' and target_rule "
        "must be null. Include a unique idea_id (UUID or short slug), title, description, "
        "rationale, and pseudocode."
    )

    idea = llm_structured(
        model=config.llm_model,
        system=_IDEA_GENERATION_SYSTEM,
        user=user,
        output_type=RuleIdea,
    )

    # There is exactly one active rule at a time, so a "fix" can only ever
    # target it — never trust an LLM-guessed target_rule string here.
    if plan.action == "fix" and plan.rule_id is not None:
        idea = idea.model_copy(update={"kind": "modify_rule", "target_rule": plan.rule_id})

    return idea
