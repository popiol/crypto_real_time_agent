"""Plan next cycle — design.md §8.2.

Evaluates the currently active rule's latest score and decides the plan for
the next cycle: 'continue' if it's performing, otherwise an LLM diagnosis of
whether the failure is fixable ('fix', with a specific change to attempt) or
fundamental ('new_rule', with a fresh direction to explore).

Also owns plan.json's rule_id/cycle_id — the record that IS the active rule
(see strategy.py's get_active_rule) — written whenever a new rule is
implemented, and read back here to know which rule to evaluate.

Whenever a rule is retired (action moves from 'continue' to 'fix' or
'new_rule'), also writes that rule's episodic trace — its final, mature
rule_evaluation.json snapshot plus the same LLM call's narrative diagnosis,
reusing one LLM call for both the fix/new_rule verdict and the trace's
diagnosis text rather than making two. This replaces writing a trace eagerly
on the first cycle after implementation (when a rule can only ever show
zero signals, since resolution takes up to 24h) — traces are meant to
capture what a rule actually did, not its birth snapshot.

Not a pipeline.py stage in its own right, and not one of the 6 numbered
steps — this logic runs at three different points across them, not once in
sequence:
  - step3_relation_analysis.py — continue re-check before deciding whether
    to bother analysing anything this cycle
  - step5_generate_idea.py — failure fallback if idea generation fails
  - step6_implement_rule.py — recording a successful implementation, or the
    failure fallback if implementation fails

Reads:
  data/state/rule_evaluation.json   — the active rule's score
  data/state/indicator_set.json     — current indicator set version (for retirement traces)
Writes:
  data/state/plan.json
  data/state/traces/<cycle_id>.json — only when a rule is retired
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.agent.models import AppConfig
from src.updater import paths
from src.updater.llm import llm_structured
from src.updater.models import EpisodicTrace, IndicatorSet, Plan, RelationAnalysis, RuleEvaluation, RuleScore
from src.updater.rule_constraints import STATELESS_CONSTRAINT

logger = logging.getLogger(__name__)

_FAILURE_DIAGNOSIS_SYSTEM = (
    "You are a trading system analyst reviewing the outcome of a deployed rule. "
    "Given performance metrics and relation analysis findings: "
    "1) write a concise diagnosis (3-5 sentences) explaining why the rule performed as it did, "
    "specific about which market conditions helped or hurt performance; "
    "2) determine whether the failure is fixable (wrong thresholds, missing conditions, "
    "parameter tuning) or fundamental (the rule direction does not work for the available data), "
    "returning 'fix' with a specific actionable change, or 'new_rule' with a new direction. "
    "Do not recommend an entry-price-relative stop-loss or profit-target as the fix. "
    + STATELESS_CONSTRAINT
)


class _CycleOutcomeDiagnosis(BaseModel):
    diagnosis: str = Field(
        description="3-5 sentence narrative explaining why the rule performed as it did"
    )
    action: Literal["fix", "new_rule"]
    description: str = Field(
        description=(
            "For 'fix': a specific, actionable description of what to change in the next version. "
            "For 'new_rule': the direction for the next relation analysis to explore."
        )
    )


def load_plan(state_dir: Path) -> Plan:
    path = paths.plan(state_dir)
    if not path.exists():
        return Plan(action="new_rule")
    try:
        return Plan.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not parse plan.json; treating as new_rule")
        return Plan(action="new_rule")


def write_implemented(state_dir: Path, rule_id: str, cycle_id: str) -> Plan:
    """Record a newly implemented rule as the active one. Nothing to evaluate
    yet, so the plan for next cycle starts at 'continue'.
    """
    prior = load_plan(state_dir)
    plan = Plan(
        rule_id=rule_id,
        cycle_id=cycle_id,
        action="continue",
        prev_action=prior.action,
        description=prior.description,
    )
    paths.plan(state_dir).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "plan.json written: rule_id=%s cycle_id=%s action=continue (just implemented)",
        rule_id,
        cycle_id,
    )
    return plan


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
        f"  avg_gain_pct: {rule_score.avg_gain_pct:.4f}  (theoretical, from signal entry/exit prices)\n"
        f"  avg_transaction_gain: {rule_score.avg_transaction_gain:.4f} "
        f"over {rule_score.transaction_count} portfolio transaction(s)  (realized, net of fees)\n"
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
    current: Plan,
    analysis: RelationAnalysis,
    cycle_id: str,
    config: AppConfig,
) -> Plan:
    """Re-evaluate current.rule_id and decide the plan for next cycle,
    preserving rule_id (the active rule doesn't change here — only
    implement_rule.py's step does that, via write_implemented). cycle_id is
    refreshed to the current cycle on every write, so plan.json always shows
    the pipeline's last-run cycle — a stale cycle_id then reliably means the
    pipeline stopped running, not just that the same rule has stayed active
    for a while. When the rule is retired (action moves to fix/new_rule),
    also writes its episodic trace, tagged with this same current cycle_id
    (the retirement moment) rather than the rule's original implementation
    cycle_id.
    """
    plan_path = paths.plan(state_dir)

    def _write(action: Literal["continue", "fix", "new_rule"], description: str | None) -> Plan:
        plan = current.model_copy(update={
            "cycle_id": cycle_id,
            "prev_action": current.action,
            "action": action,
            # Never null out the last real description — 'continue' branches
            # pass description=None here to mean "no new diagnosis", not
            # "forget the last one".
            "description": description if description is not None else current.description,
        })
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        return plan

    if current.rule_id is None:
        # First cycle: no previous rule to evaluate
        plan = _write("continue", None)
        logger.info("plan.json written: action=continue (first cycle)")
        return plan

    rule_score = get_rule_score(state_dir, current.rule_id)
    if rule_score is None:
        # No entry yet — nothing to judge.
        plan = _write("continue", None)
        logger.info(
            "plan.json written: action=continue (rule %s not yet evaluated)",
            current.rule_id,
        )
        return plan

    if rule_score.signal_count == 0 and rule_score.emitted_signal_count > 0:
        # The rule is actively firing but none of its signals have resolved
        # yet — a signal can only resolve via a matching opposite-direction
        # signal or a 24h timeout (see evaluator.py), so this is expected
        # for the first cycle or two after it starts firing, regardless of
        # how well the rule is actually performing. Treating unresolved as
        # 0% gain would kill every rule before it ever gets a fair look. If
        # it genuinely never emits any signal at all (emitted_signal_count
        # == 0), fall through instead — that's a real failure, not a timing
        # artifact.
        plan = _write("continue", None)
        logger.info(
            "plan.json written: action=continue (rule %s has %d emitted "
            "signal(s), none evaluated yet)",
            current.rule_id,
            rule_score.emitted_signal_count,
        )
        return plan

    metric_value, metric_name = _performance_metric(rule_score)
    if metric_value > config.cycle_success_threshold:
        plan = _write("continue", None)
        logger.info(
            "plan.json written: action=continue (rule %s %s=%.4f > threshold)",
            current.rule_id,
            metric_name,
            metric_value,
        )
        return plan

    # Failure: ask LLM to diagnose fix vs new_rule, and retire the rule with
    # a trace capturing its final, mature performance.
    try:
        diagnosis = _diagnose_failure(rule_score, analysis, config)
        plan = _write(diagnosis.action, diagnosis.description)
        _write_trace(state_dir, rule_score, diagnosis.diagnosis, cycle_id, config)
    except Exception:
        logger.exception("Failure diagnosis LLM call failed; defaulting to new_rule")
        fallback_diagnosis = f"No diagnosis available for {current.rule_id}."
        plan = _write(
            "new_rule",
            f"Rule {current.rule_id} underperformed ({metric_name}={metric_value:.4f}); explore a new direction.",
        )
        _write_trace(state_dir, rule_score, fallback_diagnosis, cycle_id, config)

    logger.info(
        "plan.json written: action=%s (rule %s %s=%.4f) — %s",
        plan.action,
        current.rule_id,
        metric_name,
        metric_value,
        plan.description,
    )
    return plan


def _performance_metric(rule_score: RuleScore) -> tuple[float, str]:
    """Return (value, metric_name) to judge a rule's performance by."""
    return rule_score.recent_avg_gain_pct, "recent_avg_gain_pct"


# ── Episodic trace (written only when a rule is retired) ───────────────────────


def _write_trace(
    state_dir: Path,
    rule_score: RuleScore,
    diagnosis: str,
    cycle_id: str,
    config: AppConfig,
) -> None:
    traces_dir = paths.traces_dir(state_dir)
    traces_dir.mkdir(exist_ok=True)

    trace_path = paths.trace_file(state_dir, cycle_id)
    if trace_path.exists():
        logger.info("Trace for cycle %s already exists; skipping", cycle_id)
        return

    hypothesis = rule_score.description
    trace = EpisodicTrace(
        trace_id=str(uuid.uuid4()),
        cycle_id=cycle_id,
        hypothesis=hypothesis,
        rule_id=rule_score.rule_id,
        indicator_names=_load_indicator_names(state_dir),
        outcome_metrics={
            "avg_gain_pct": rule_score.avg_gain_pct,
            "positive_rate": rule_score.positive_rate,
            "min_gain_pct": rule_score.min_gain_pct,
            "p25_gain_pct": rule_score.p25_gain_pct,
            "p75_gain_pct": rule_score.p75_gain_pct,
            "signal_count": float(rule_score.signal_count),
            "score": rule_score.score,
        },
        diagnosis=diagnosis,
        embedding=_compute_embedding(hypothesis, config.embedding_model),
    )
    trace_path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Episodic trace written: %s (rule=%s, retired)", trace_path, rule_score.rule_id)


def _load_indicator_names(state_dir: Path) -> list[str]:
    path = paths.indicator_set(state_dir)
    if not path.exists():
        return []
    try:
        indicator_set = IndicatorSet.model_validate_json(path.read_text(encoding="utf-8"))
        return [i.name for i in indicator_set.indicators]
    except Exception:
        return []


def _compute_embedding(text: str, embedding_model: str) -> list[float]:
    try:
        if embedding_model.startswith("models/") or "embedding" in embedding_model:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embedder = GoogleGenerativeAIEmbeddings(model=embedding_model)
            return embedder.embed_query(text)
        logger.warning("Unsupported embedding model '%s'; storing empty embedding", embedding_model)
        return []
    except Exception:
        logger.warning("Embedding computation failed", exc_info=True)
        return []
