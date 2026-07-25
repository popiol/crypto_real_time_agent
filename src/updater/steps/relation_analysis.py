"""Relation analysis — design.md §8.2 Step 3.

Retrieves the top-K episodic traces most semantically similar to the current
cycle plan's description, then makes one LLM call that looks at this cycle's
computed indicator values, a sample of the accumulated train set, and those
retrieved traces to identify which indicator patterns preceded positive vs.
negative outcomes.

Not a pipeline.py stage in its own right: its output (RelationAnalysis) is
only ever consumed in-memory by generate_idea.py in the same run, so this
module is called directly from step7_implement_idea.py's orchestrator.

Reads:
  data/state/indicator_values.json  — computed indicator values (from step2_compute_indicators)
  data/state/train_set.json         — accumulated train samples
  data/state/traces/                — episodic trace files
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent.models import AppConfig
from src.updater import paths
from src.updater.llm import llm_structured
from src.updater.models import EpisodicTrace, RelationAnalysis, TrainSet

logger = logging.getLogger(__name__)

_MAX_TRAIN_SAMPLES = 100  # cap sent to LLM

_RELATION_ANALYSIS_SYSTEM = (
    "You are a quantitative trading analyst. "
    "You will receive indicator values computed for recent signals, "
    "a sample of the training dataset linking indicator values to price outcomes, "
    "and relevant past episodic traces describing previously tested hypotheses. "
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


def _load_indicator_values(state_dir: Path) -> dict[str, dict[str, float | None]]:
    path = paths.indicator_values(state_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read indicator_values.json", exc_info=True)
        return {}


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


def _format_indicator_values(values: dict[str, dict[str, float | None]]) -> str:
    if not values:
        return "(no indicator values available)"
    lines = []
    for pair, indicators in values.items():
        row = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}=null"
            for k, v in indicators.items()
        )
        lines.append(f"  {pair}: {row}")
    return "\n".join(lines)


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
    indicator_values = _load_indicator_values(state_dir)
    train_samples = _load_recent_train_samples(state_dir, _MAX_TRAIN_SAMPLES)

    indicators_section = _format_indicator_values(indicator_values)
    traces_section = _format_traces(traces)

    if train_samples:
        train_section = json.dumps(train_samples, indent=2)
    else:
        train_section = "(no training samples available yet)"

    user = (
        f"Current indicator values by pair:\n{indicators_section}\n\n"
        f"Recent training samples (last {len(train_samples)}):\n{train_section}\n\n"
        f"Top-{config.trace_top_k} relevant past traces:\n{traces_section}"
    )

    return llm_structured(
        model=config.llm_model,
        system=_RELATION_ANALYSIS_SYSTEM,
        user=user,
        output_type=RelationAnalysis,
    )
