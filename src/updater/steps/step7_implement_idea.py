"""Step 7 — orchestrates relation analysis, idea generation, rule
implementation, and cycle planning (design.md §8.2 Steps 3, 4, 5, and 9).

Those four steps are not split into independent pipeline.py stages: per
design.md, their outputs (RelationAnalysis, RuleIdea) are explicitly
in-memory, consumed by the next step in the same run rather than persisted
and independently resumable — unlike e.g. rule_evaluation.json or
train_set.json. This module is the single pipeline.py entry point that
threads them together; the actual logic for each lives in its own module:

  relation_analysis.py  — design.md Step 3
  generate_idea.py      — design.md Step 4
  implement_rule.py     — design.md Step 5
  plan_next_cycle.py     — design.md Step 9 (also owns last_implemented.json)

Rule folder layout:
  src/strategy/rules/<rule_name>/v1.py   ← initial version
  src/strategy/rules/<rule_name>/v2.py   ← revised version, etc.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from src.agent.models import AppConfig
from src.updater.models import RelationAnalysis
from src.updater.steps import generate_idea as generate_idea_step
from src.updater.steps import implement_rule as implement_rule_step
from src.updater.steps import plan_next_cycle as plan_next_cycle_step
from src.updater.steps import relation_analysis as relation_analysis_step

logger = logging.getLogger(__name__)

_EMPTY_ANALYSIS = RelationAnalysis(
    positive_patterns=[], negative_patterns=[], key_indicators=[], suggested_direction=""
)


def run(config: AppConfig, state_dir: Path, cycle_id: str) -> None:
    # Load previous cycle's plan and context
    current_plan = plan_next_cycle_step.load_current_plan(state_dir)
    last_rule_id, _last_cycle_id = plan_next_cycle_step.load_last_implemented(state_dir)

    if current_plan.action == "continue":
        # The active rule is performing — exactly one rule can ever be active
        # (see strategy.py), so generating and implementing a new idea here
        # would replace a winning rule with an untested one. Just re-check
        # whether it's still performing, and leave it running otherwise.
        logger.info(
            "Cycle plan: action=continue — rule %s is performing; skipping idea generation",
            last_rule_id,
        )
        plan_next_cycle_step.write_next_cycle_plan(
            state_dir, last_rule_id, _EMPTY_ANALYSIS, config
        )
        return

    # Retrieve relevant past traces and run relation analysis
    traces = relation_analysis_step.retrieve_top_k_traces(
        state_dir, current_plan.description, config
    )
    logger.info(
        "Cycle plan: action=%s%s | retrieved %d trace(s)",
        current_plan.action,
        f" — {current_plan.description}" if current_plan.description else "",
        len(traces),
    )
    try:
        analysis = relation_analysis_step.run_relation_analysis(state_dir, traces, config)
        logger.info(
            "Relation analysis: %d positive pattern(s), %d negative pattern(s), "
            "key_indicators=%s | suggested_direction=%s",
            len(analysis.positive_patterns),
            len(analysis.negative_patterns),
            analysis.key_indicators,
            analysis.suggested_direction,
        )
    except Exception:
        logger.exception("Relation analysis failed; using empty analysis")
        analysis = RelationAnalysis(
            positive_patterns=[],
            negative_patterns=[],
            key_indicators=[],
            suggested_direction="No analysis available — explore a new rule direction.",
        )

    # Generate exactly one rule idea
    try:
        idea = generate_idea_step.generate_idea(analysis, current_plan, last_rule_id, config)
        # Ensure idea_id is set
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
        logger.exception("Idea generation failed; skipping implementation")
        plan_next_cycle_step.write_next_cycle_plan(state_dir, last_rule_id, analysis, config)
        return

    # Implement the idea
    rule_id, rule_path = implement_rule_step.next_rule_path(idea)
    implemented_rule_id: str | None = None
    try:
        implemented = implement_rule_step.generate_code(idea, rule_id, config.llm_model)
        rule_path.parent.mkdir(parents=True, exist_ok=True)
        rule_path.write_text(implemented.code, encoding="utf-8")
        implement_rule_step.commit_and_push(rule_id, implemented.function_name)
        implemented_rule_id = rule_id
        logger.info("Implemented rule %s at %s", rule_id, rule_path)
    except implement_rule_step.ImplementationFailed:
        logger.exception("Rule implementation failed for idea '%s'", idea.title)

    # Write last_implemented.json for the new rule
    if implemented_rule_id:
        plan_next_cycle_step.write_last_implemented(state_dir, implemented_rule_id, cycle_id)

    # Write next_cycle_plan.json based on previous rule's performance
    plan_next_cycle_step.write_next_cycle_plan(state_dir, last_rule_id, analysis, config)
