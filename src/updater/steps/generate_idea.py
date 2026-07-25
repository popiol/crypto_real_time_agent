"""Generate rule idea — design.md §8.2 Step 4.

Given the relation analysis findings and the current cycle plan, generates
exactly one RuleIdea. For a 'fix' plan, target_rule is always the current
active rule (taken from last_implemented.json) — never left to the LLM to
guess, since exactly one rule is ever active (see strategy.py).

Not a pipeline.py stage in its own right: it consumes the RelationAnalysis
produced in-memory by relation_analysis.py in the same run, so this module
is called directly from step7_implement_idea.py's orchestrator.
"""

from __future__ import annotations

import logging

from src.agent.models import AppConfig
from src.updater.llm import llm_structured
from src.updater.models import NextCyclePlan, RelationAnalysis, RuleIdea
from src.updater.steps.implement_rule import DATA_WINDOW_CONSTRAINT

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


def generate_idea(
    analysis: RelationAnalysis,
    current_plan: NextCyclePlan,
    last_rule_id: str | None,
    config: AppConfig,
) -> RuleIdea:
    plan_section = f"Current cycle plan:\nAction: {current_plan.action}\n"
    if current_plan.description:
        plan_section += f"Description: {current_plan.description}\n"

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
    if current_plan.action == "fix" and last_rule_id is not None:
        idea = idea.model_copy(update={"kind": "modify_rule", "target_rule": last_rule_id})

    return idea
