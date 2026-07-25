"""Plan next cycle — design.md §8.2 Step 9.

Evaluates the currently active rule's latest score and decides the plan for
the next cycle: 'continue' if it's performing, otherwise an LLM diagnosis of
whether the failure is fixable ('fix', with a specific change to attempt) or
fundamental ('new_rule', with a fresh direction to explore).

Also owns last_implemented.json, the {rule_id, cycle_id} record that IS the
active rule (see strategy.py's get_active_rule) — written whenever a new rule
is implemented, and read back here to know which rule to evaluate.

Not a pipeline.py stage in its own right: it consumes the RelationAnalysis
produced in-memory by relation_analysis.py in the same run (used only for
extra context in the failure diagnosis), so this module is called directly
from step7_implement_idea.py's orchestrator.

Reads:
  data/state/rule_evaluation.json   — the active rule's score
Writes:
  data/state/last_implemented.json
  data/state/next_cycle_plan.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.agent.models import AppConfig
from src.updater import paths
from src.updater.llm import llm_structured
from src.updater.models import NextCyclePlan, RelationAnalysis, RuleEvaluation, RuleScore

logger = logging.getLogger(__name__)

_FAILURE_DIAGNOSIS_SYSTEM = (
    "You are a trading system analyst reviewing the outcome of a deployed rule. "
    "Given performance metrics and relation analysis findings, determine whether "
    "the failure is fixable (wrong thresholds, missing conditions, parameter tuning) "
    "or fundamental (the rule direction does not work for the available data). "
    "Return 'fix' with a specific actionable change, or 'new_rule' with a new direction."
)


class _CycleOutcomeDiagnosis(BaseModel):
    action: Literal["fix", "new_rule"]
    description: str = Field(
        description=(
            "For 'fix': a specific, actionable description of what to change in the next version. "
            "For 'new_rule': the direction for the next relation analysis to explore."
        )
    )


def load_current_plan(state_dir: Path) -> NextCyclePlan:
    path = paths.next_cycle_plan(state_dir)
    if not path.exists():
        return NextCyclePlan(action="new_rule", description=None)
    try:
        return NextCyclePlan.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not parse next_cycle_plan.json; treating as new_rule")
        return NextCyclePlan(action="new_rule", description=None)


def load_last_implemented(state_dir: Path) -> tuple[str | None, str | None]:
    path = paths.last_implemented(state_dir)
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("rule_id"), data.get("cycle_id")
    except Exception:
        logger.warning("Could not read last_implemented.json", exc_info=True)
        return None, None


def write_last_implemented(state_dir: Path, rule_id: str, cycle_id: str) -> None:
    path = paths.last_implemented(state_dir)
    path.write_text(
        json.dumps({"rule_id": rule_id, "cycle_id": cycle_id}, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "last_implemented.json written: rule_id=%s cycle_id=%s", rule_id, cycle_id
    )


def get_rule_score(state_dir: Path, rule_id: str) -> RuleScore | None:
    path = paths.rule_evaluation(state_dir)
    if not path.exists():
        return None
    try:
        evaluation = RuleEvaluation.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        return next((r for r in evaluation.rules if r.rule_id == rule_id), None)
    except Exception:
        logger.warning("Could not read rule_evaluation.json", exc_info=True)
        return None


def _diagnose_failure(
    rule_score: RuleScore, analysis: RelationAnalysis, config: AppConfig
) -> _CycleOutcomeDiagnosis:
    user = (
        f"Failed rule: {rule_score.rule_id}\n"
        f"Description: {rule_score.description}\n\n"
        f"Performance metrics:\n"
        f"  avg_gain_pct: {rule_score.avg_gain_pct:.4f}\n"
        f"  positive_rate: {rule_score.positive_rate:.3f}\n"
        f"  p25/p75: {rule_score.p25_gain_pct:.4f} / {rule_score.p75_gain_pct:.4f}\n"
        f"  min_gain_pct: {rule_score.min_gain_pct:.4f}\n"
        f"  signal_count: {rule_score.signal_count}\n"
        f"  signal_trend: {rule_score.signal_trend}\n"
        f"  score: {rule_score.score:.4f}\n\n"
        f"Relation analysis from this cycle:\n{analysis.model_dump_json(indent=2)}"
    )
    return llm_structured(
        model=config.llm_model,
        system=_FAILURE_DIAGNOSIS_SYSTEM,
        user=user,
        output_type=_CycleOutcomeDiagnosis,
    )


def write_next_cycle_plan(
    state_dir: Path,
    last_rule_id: str | None,
    analysis: RelationAnalysis,
    config: AppConfig,
) -> None:
    plan_path = paths.next_cycle_plan(state_dir)

    if last_rule_id is None:
        # First cycle: no previous rule to evaluate
        plan = NextCyclePlan(action="continue", description=None)
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        logger.info("next_cycle_plan.json written: action=continue (first cycle)")
        return

    rule_score = get_rule_score(state_dir, last_rule_id)
    if rule_score is None:
        # Rule not yet in evaluation (too new) — treat as continue
        plan = NextCyclePlan(action="continue", description=None)
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "next_cycle_plan.json written: action=continue (rule %s not yet evaluated)",
            last_rule_id,
        )
        return

    if rule_score.avg_gain_pct > config.cycle_success_threshold:
        plan = NextCyclePlan(action="continue", description=None)
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "next_cycle_plan.json written: action=continue (rule %s avg_gain=%.4f > threshold)",
            last_rule_id,
            rule_score.avg_gain_pct,
        )
        return

    # Failure: ask LLM to diagnose fix vs new_rule
    try:
        diagnosis = _diagnose_failure(rule_score, analysis, config)
        plan = NextCyclePlan(action=diagnosis.action, description=diagnosis.description)
    except Exception:
        logger.exception("Failure diagnosis LLM call failed; defaulting to new_rule")
        plan = NextCyclePlan(
            action="new_rule",
            description=f"Rule {last_rule_id} underperformed (avg_gain={rule_score.avg_gain_pct:.4f}); explore a new direction.",
        )

    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "next_cycle_plan.json written: action=%s (rule %s avg_gain=%.4f) — %s",
        plan.action,
        last_rule_id,
        rule_score.avg_gain_pct,
        plan.description,
    )
