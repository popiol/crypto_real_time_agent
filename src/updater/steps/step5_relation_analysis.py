"""Relation analysis — design.md §8.2 Step 5.

run() is this pipeline.py stage's entry point: it re-checks the current plan
fresh (rather than trusting the previous cycle's stale verdict); if the
active rule is still performing, it stops there and nothing downstream runs
this cycle. Otherwise it retrieves the top-K episodic traces most
semantically similar to the current plan's description, then makes one LLM
call that looks at a sample of the accumulated train set (indicator values
paired with their resolved outcome) and those traces to identify which
indicator patterns preceded positive vs. negative outcomes — and persists the
result for step6_generate_idea.py's step to pick up.

Reads:
  data/state/plan.json, data/state/train_set.json, data/state/traces/
Writes:
  data/state/plan.json (re-checked), data/state/relation_analysis.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent.models import AppConfig
from src.updater import paths
from src.updater import plan_next_cycle as plan_next_cycle_step
from src.updater.llm import llm_structured
from src.updater.models import EpisodicTrace, PendingRelationAnalysis, RelationAnalysis, TrainSet

logger = logging.getLogger(__name__)

_MAX_TRAIN_SAMPLES = 100  # cap sent to LLM

_EMPTY_ANALYSIS = RelationAnalysis(
    positive_patterns=[], negative_patterns=[], key_indicators=[], suggested_direction=""
)

_RELATION_ANALYSIS_SYSTEM = (
    "You are a quantitative trading analyst. "
    "You will receive a sample of the training dataset linking indicator values to "
    "price outcomes, and relevant past episodic traces describing previously tested "
    "hypotheses. "
    "Identify which indicator combinations or conditions preceded positive outcomes "
    "and which preceded losses. Be specific about directions and thresholds. "
    "Your analysis will directly inform the generation of a new trading rule."
)


# ── Trace retrieval ───────────────────────────────────────────────────────────


def _compute_embedding(text: str, embedding_model: str) -> list[float]:
    try:
        if embedding_model.startswith("models/") or "embedding" in embedding_model:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embedder = GoogleGenerativeAIEmbeddings(model=embedding_model)
            return embedder.embed_query(text)
    except Exception:
        logger.warning("Embedding computation failed", exc_info=True)
    return []


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    try:
        import numpy as np

        va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)
    except Exception:
        return 0.0


def retrieve_top_k_traces(
    state_dir: Path, query_text: str | None, config: AppConfig
) -> list[EpisodicTrace]:
    traces_dir = paths.traces_dir(state_dir)
    if not traces_dir.exists():
        return []

    trace_files = sorted(traces_dir.glob("*.json"))
    if not trace_files:
        return []

    k = config.trace_top_k

    # If no query text, return K most recent traces (by filename, which is timestamp-based)
    if not query_text:
        recent = trace_files[-k:]
        result = []
        for f in recent:
            try:
                result.append(
                    EpisodicTrace.model_validate_json(f.read_text(encoding="utf-8"))
                )
            except Exception:
                logger.warning("Could not parse trace %s", f, exc_info=True)
        return result

    # Embed query and score traces
    query_embedding = _compute_embedding(query_text, config.embedding_model)
    if not query_embedding:
        # Embedding failed; fall back to K most recent
        recent = trace_files[-k:]
        result = []
        for f in recent:
            try:
                result.append(
                    EpisodicTrace.model_validate_json(f.read_text(encoding="utf-8"))
                )
            except Exception:
                logger.warning("Could not parse trace %s", f, exc_info=True)
        return result

    scored: list[tuple[float, EpisodicTrace]] = []
    for f in trace_files:
        try:
            trace = EpisodicTrace.model_validate_json(f.read_text(encoding="utf-8"))
            if not trace.embedding:
                continue
            sim = _cosine_similarity(query_embedding, trace.embedding)
            scored.append((sim, trace))
        except Exception:
            logger.warning("Could not load/score trace %s", f, exc_info=True)

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]


# ── Relation analysis ─────────────────────────────────────────────────────────


def _load_recent_train_samples(state_dir: Path, n: int) -> list[dict]:
    path = paths.train_set(state_dir)
    if not path.exists():
        return []
    try:
        train_set = TrainSet.model_validate_json(path.read_text(encoding="utf-8"))
        samples = train_set.samples[-n:]
        return [s.model_dump() for s in samples]
    except Exception:
        logger.warning("Could not read train_set.json", exc_info=True)
        return []


def _format_traces(traces: list[EpisodicTrace]) -> str:
    if not traces:
        return "(no relevant past traces)"
    parts = []
    for t in traces:
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in t.outcome_metrics.items())
        parts.append(
            f"Trace {t.cycle_id} (rule={t.rule_id}):\n"
            f"  Hypothesis: {t.hypothesis}\n"
            f"  Metrics: {metrics}\n"
            f"  Diagnosis: {t.diagnosis}"
        )
    return "\n\n".join(parts)


def run_relation_analysis(
    state_dir: Path, traces: list[EpisodicTrace], config: AppConfig
) -> RelationAnalysis:
    train_samples = _load_recent_train_samples(state_dir, _MAX_TRAIN_SAMPLES)

    traces_section = _format_traces(traces)

    if train_samples:
        train_section = json.dumps(train_samples, indent=2)
    else:
        train_section = "(no training samples available yet)"

    user = (
        f"Recent training samples (last {len(train_samples)}):\n{train_section}\n\n"
        f"Top-{config.trace_top_k} relevant past traces:\n{traces_section}"
    )

    return llm_structured(
        model=config.llm_model,
        system=_RELATION_ANALYSIS_SYSTEM,
        user=user,
        output_type=RelationAnalysis,
    )


def run(config: AppConfig, state_dir: Path, cycle_id: str) -> None:
    plan = plan_next_cycle_step.load_plan(state_dir)

    if plan.action == "continue":
        # The plan on disk was decided last cycle — re-check fresh rather than
        # trust it blindly, since the active rule's score may have moved on.
        plan = plan_next_cycle_step.write_next_cycle_plan(
            state_dir, plan, _EMPTY_ANALYSIS, config
        )
        if plan.action == "continue":
            # Exactly one rule can ever be active (see strategy.py), so
            # running relation analysis here would only ever feed an idea
            # that proposes replacing a winning rule with an untested one —
            # leave it running and skip everything downstream this cycle.
            logger.info(
                "Cycle plan: action=continue — rule %s is performing; skipping relation analysis",
                plan.rule_id,
            )
            return
        logger.info(
            "Rule %s no longer performing (re-checked action=%s); running relation "
            "analysis this cycle instead of waiting for the next one",
            plan.rule_id,
            plan.action,
        )

    traces = retrieve_top_k_traces(state_dir, plan.description, config)
    logger.info(
        "Cycle plan: action=%s%s | retrieved %d trace(s)",
        plan.action,
        f" — {plan.description}" if plan.description else "",
        len(traces),
    )
    try:
        analysis = run_relation_analysis(state_dir, traces, config)
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

    pending = PendingRelationAnalysis(cycle_id=cycle_id, plan=plan, analysis=analysis)
    paths.relation_analysis(state_dir).write_text(
        pending.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("relation_analysis.json written (cycle_id=%s)", cycle_id)
