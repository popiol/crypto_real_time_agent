"""Episodic trace — design.md §8.2 Step 4. Writes an immutable trace record
for the current cycle.

Traces capture the hypothesis (rule description), outcome metrics, and an
LLM-generated diagnosis for the most recently evaluated rule. The hypothesis
text is embedded and stored alongside the trace for future semantic retrieval.

Each trace is written to data/state/traces/<cycle_id>.json. Existing files
are never overwritten.

Reads:
  data/state/plan.json              — which rule to trace (rule_id/cycle_id, written by implement step)
  data/state/rule_evaluation.json   — outcome metrics for that rule
  data/state/indicator_set.json     — current indicator set version

Writes:
  data/state/traces/<cycle_id>.json
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from pydantic import BaseModel

from src.agent.models import AppConfig
from src.updater import paths
from src.updater.llm import llm_structured
from src.updater.models import EpisodicTrace, IndicatorSet, RuleEvaluation, RuleScore

logger = logging.getLogger(__name__)

_DIAGNOSIS_SYSTEM = (
    "You are a quantitative trading analyst. "
    "Given the performance metrics of a trading rule and the indicators it used, "
    "write a concise diagnosis (3-5 sentences) explaining why the rule performed as it did. "
    "Be specific about which market conditions helped or hurt performance."
)


class _Diagnosis(BaseModel):
    diagnosis: str


def run(config: AppConfig, state_dir: Path, cycle_id: str) -> None:
    traces_dir = paths.traces_dir(state_dir)
    traces_dir.mkdir(exist_ok=True)

    rule_id, trace_cycle_id = _resolve_target_rule(state_dir, cycle_id)
    if rule_id is None:
        logger.info("No implemented rule to trace; skipping episodic trace step")
        return

    trace_path = paths.trace_file(state_dir, trace_cycle_id)
    if trace_path.exists():
        logger.info("Trace for cycle %s already exists; skipping", trace_cycle_id)
        return

    rule_score = _load_rule_score(state_dir, rule_id)
    if rule_score is None:
        logger.info("Rule %s not found in rule_evaluation.json; skipping trace", rule_id)
        return

    indicator_set_version = _load_indicator_version(state_dir)
    hypothesis = rule_score.description

    diagnosis = _generate_diagnosis(rule_score, config.llm_model)
    embedding = _compute_embedding(hypothesis, config.embedding_model)

    outcome_metrics = {
        "avg_gain_pct": rule_score.avg_gain_pct,
        "positive_rate": rule_score.positive_rate,
        "min_gain_pct": rule_score.min_gain_pct,
        "p25_gain_pct": rule_score.p25_gain_pct,
        "p75_gain_pct": rule_score.p75_gain_pct,
        "signal_count": float(rule_score.signal_count),
        "score": rule_score.score,
    }

    trace = EpisodicTrace(
        trace_id=str(uuid.uuid4()),
        cycle_id=trace_cycle_id,
        hypothesis=hypothesis,
        rule_id=rule_id,
        indicator_set_version=indicator_set_version,
        outcome_metrics=outcome_metrics,
        diagnosis=diagnosis,
        embedding=embedding,
    )

    trace_path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Episodic trace written: %s (rule=%s)", trace_path, rule_id)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_target_rule(
    state_dir: Path, cycle_id: str
) -> tuple[str | None, str]:
    """Return (rule_id, cycle_id) for the rule to trace.

    Uses the cycle_id recorded in plan.json — the cycle that actually
    implemented this rule, which is always written alongside rule_id (see
    plan_next_cycle.write_implemented). plan.json is the sole source of
    truth for which rule is active (see strategy.py's get_active_rule): if
    it's missing, rule_evaluation.json can't hold a trustworthy answer
    either, since step2_evaluate_rule only ever scores whatever this file
    says is active.
    """
    plan_path = paths.plan(state_dir)
    if plan_path.exists():
        try:
            data = json.loads(plan_path.read_text(encoding="utf-8"))
            if data.get("rule_id") is not None:
                return data["rule_id"], data["cycle_id"]
        except Exception:
            logger.warning("Could not read plan.json", exc_info=True)

    return None, cycle_id


def _load_rule_score(state_dir: Path, rule_id: str) -> RuleScore | None:
    path = paths.rule_evaluation(state_dir)
    if not path.exists():
        return None
    try:
        evaluation = RuleEvaluation.model_validate_json(path.read_text(encoding="utf-8"))
        return next((r for r in evaluation.rules if r.rule_id == rule_id), None)
    except Exception:
        logger.warning("Could not read rule_evaluation.json", exc_info=True)
        return None


def _load_indicator_version(state_dir: Path) -> str:
    path = paths.indicator_set(state_dir)
    if not path.exists():
        return "none"
    try:
        indicator_set = IndicatorSet.model_validate_json(path.read_text(encoding="utf-8"))
        return indicator_set.version
    except Exception:
        return "unknown"


def _generate_diagnosis(rule_score: RuleScore, model: str) -> str:
    try:
        result = llm_structured(
            model=model,
            system=_DIAGNOSIS_SYSTEM,
            user=(
                f"Rule: {rule_score.rule_id}\n"
                f"Description: {rule_score.description}\n\n"
                f"Performance metrics:\n"
                f"  avg_gain_pct: {rule_score.avg_gain_pct:.4f}\n"
                f"  positive_rate: {rule_score.positive_rate:.3f}\n"
                f"  p25/p75: {rule_score.p25_gain_pct:.4f} / {rule_score.p75_gain_pct:.4f}\n"
                f"  min_gain_pct: {rule_score.min_gain_pct:.4f}\n"
                f"  signal_count: {rule_score.signal_count}\n"
                f"  signal_trend: {rule_score.signal_trend}\n"
                f"  score: {rule_score.score:.4f}\n"
            ),
            output_type=_Diagnosis,
        )
        return result.diagnosis
    except Exception:
        logger.warning("LLM diagnosis call failed for rule %s", rule_score.rule_id, exc_info=True)
        return f"No diagnosis available for {rule_score.rule_id}."


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
