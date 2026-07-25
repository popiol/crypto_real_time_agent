"""Step 8 — Relation analysis, idea generation, rule implementation, and cycle planning.

Pipeline steps carried out here:
  1. Unregister deprecated rule versions from strategy.py.
  2. Read current next_cycle_plan.json (written by previous cycle).
  3. Retrieve top-K episodic traces via cosine similarity to plan description.
  4. Run LLM relation analysis (indicator values + train set + retrieved traces).
  5. Generate exactly one RuleIdea from the analysis.
  6. Generate and register rule code (new_rule → new folder/v1.py; fix → new version).
  7. Write last_implemented.json with new rule_id and cycle_id.
  8. Evaluate previous rule's performance and write next_cycle_plan.json for next cycle.

Rule folder layout:
  src/strategy/rules/<rule_name>/v1.py   ← initial version
  src/strategy/rules/<rule_name>/v2.py   ← revised version, etc.

Reads:
  data/state/next_cycle_plan.json   — current cycle's plan (from previous cycle)
  data/state/last_implemented.json  — previous cycle's rule_id (to evaluate performance)
  data/state/rule_evaluation.json   — all rule scores
  data/state/indicator_values.json  — computed indicator values (from step 2)
  data/state/train_set.json         — accumulated train samples
  data/state/traces/                — episodic trace files

Writes:
  src/strategy/rules/<rule_name>/v<N>.py
  src/strategy/strategy.py
  data/state/last_implemented.json
  data/state/next_cycle_plan.json
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import re
import subprocess
import uuid
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import src.agent.models as _agent_models
from src.agent import storage
from src.agent.models import AppConfig
from src.updater import paths
from src.updater.code_diff import CodeDiff, apply_changes
from src.updater.llm import llm_structured, make_llm
from src.updater.models import (
    EpisodicTrace,
    ImplementedRule,
    NextCyclePlan,
    RuleEvaluation,
    RuleIdea,
    RuleScore,
    TrainSet,
)

logger = logging.getLogger(__name__)

_MODELS_SOURCE = (
    "\n\n".join(
        inspect.getsource(cls)
        for cls in (
            _agent_models.Tick,
            _agent_models.WarmCandle,
            _agent_models.ColdMonth,
            _agent_models.PairData,
            _agent_models.BuySignal,
            _agent_models.SellSignal,
        )
    )
    + "\n\nMarketData = dict[str, PairData]"
)

_RULES_DIR = Path("src/strategy/rules")

_MAX_FIX_ATTEMPTS = 10
_MAX_TRAIN_SAMPLES = 100  # cap sent to LLM


# ── Local models ─────────────────────────────────────────────────────────────


class _RelationAnalysis(BaseModel):
    positive_patterns: list[str] = Field(
        description="Indicator combinations or conditions that correlated with positive outcomes"
    )
    negative_patterns: list[str] = Field(
        description="Indicator combinations or conditions that correlated with negative outcomes"
    )
    key_indicators: list[str] = Field(
        description="Most informative indicators for predicting outcome sign"
    )
    suggested_direction: str = Field(
        description="Proposed direction for the next rule idea, grounded in the observed patterns"
    )


class _CycleOutcomeDiagnosis(BaseModel):
    action: Literal["fix", "new_rule"]
    description: str = Field(
        description=(
            "For 'fix': a specific, actionable description of what to change in the next version. "
            "For 'new_rule': the direction for the next relation analysis to explore."
        )
    )


class _ImplementationFailed(Exception):
    pass


# ── System prompts ────────────────────────────────────────────────────────────

_RELATION_ANALYSIS_SYSTEM = (
    "You are a quantitative trading analyst. "
    "You will receive indicator values computed for recent signals, "
    "a sample of the training dataset linking indicator values to price outcomes, "
    "and relevant past episodic traces describing previously tested hypotheses. "
    "Identify which indicator combinations or conditions preceded positive outcomes "
    "and which preceded losses. Be specific about directions and thresholds. "
    "Your analysis will directly inform the generation of a new trading rule."
)

_IDEA_GENERATION_SYSTEM = (
    "You are a quantitative trading strategist. "
    "Based on the relation analysis findings and the current cycle plan, "
    "generate exactly ONE concrete trading rule idea. "
    "The idea must be directly grounded in the observed indicator patterns. "
    "For 'fix', target the named rule and describe specific improvements. "
    "For 'new_rule', propose a genuinely different approach. "
    "Be precise: include thresholds, conditions, and expected market behaviour."
)

_FAILURE_DIAGNOSIS_SYSTEM = (
    "You are a trading system analyst reviewing the outcome of a deployed rule. "
    "Given performance metrics and relation analysis findings, determine whether "
    "the failure is fixable (wrong thresholds, missing conditions, parameter tuning) "
    "or fundamental (the rule direction does not work for the available data). "
    "Return 'fix' with a specific actionable change, or 'new_rule' with a new direction."
)

_IMPLEMENT_SYSTEM = (
    "You are an expert Python developer specialising in quantitative trading rules. "
    "Generate a complete, self-contained Python module that implements the described rule. "
    "Return ONLY the raw Python source code — no explanation, no markdown, no code fences."
)

_FIX_SYSTEM = (
    "You are an expert Python developer specialising in quantitative trading rules. "
    "You will be given a partial or broken implementation of a trading rule. "
    "Return the changes needed to complete it into a fully working module. "
    "Do not rewrite parts that are already correct."
)

_REFERENCE_RULE = """\
\"\"\"Rule 01 — Spread compression spike (v1).\"\"\"
from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal

MIN_TICKS = 10
COMPRESSION_THRESHOLD = 0.30

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < MIN_TICKS:
            continue
        spreads = [t.spread_rel for t in ticks]
        baseline = statistics.mean(spreads[:-1])
        current = spreads[-1]
        if baseline > 0 and current < baseline * (1 - COMPRESSION_THRESHOLD):
            signals.append(BuySignal(
                pair=pair,
                timestamp=ticks[-1].polled_at,
                price=ticks[-1].last_price,
            ))
        elif baseline > 0 and current > baseline * (1 + COMPRESSION_THRESHOLD):
            signals.append(SellSignal(
                pair=pair,
                timestamp=ticks[-1].polled_at,
                price=ticks[-1].last_price,
            ))
    return signals
"""


# ── Main run ──────────────────────────────────────────────────────────────────


def run(config: AppConfig, state_dir: Path) -> None:
    # Load previous cycle's plan and context
    current_plan = _load_current_plan(state_dir)
    last_rule_id, _last_cycle_id = _load_last_implemented(state_dir)

    # Retrieve relevant past traces and run relation analysis
    traces = _retrieve_top_k_traces(state_dir, current_plan.description, config)
    logger.info(
        "Cycle plan: action=%s%s | retrieved %d trace(s)",
        current_plan.action,
        f" — {current_plan.description}" if current_plan.description else "",
        len(traces),
    )
    try:
        analysis = _run_relation_analysis(state_dir, traces, config)
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
        analysis = _RelationAnalysis(
            positive_patterns=[],
            negative_patterns=[],
            key_indicators=[],
            suggested_direction="No analysis available — explore a new rule direction.",
        )

    # Generate exactly one rule idea
    try:
        idea = _generate_idea(analysis, current_plan, last_rule_id, config)
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
        _write_next_cycle_plan(state_dir, last_rule_id, analysis, config)
        return

    # Implement the idea
    rule_id, rule_path = _next_rule_path(idea)
    implemented_rule_id: str | None = None
    try:
        implemented = _generate_code(idea, rule_id, config.llm_model)
        rule_path.parent.mkdir(parents=True, exist_ok=True)
        rule_path.write_text(implemented.code, encoding="utf-8")
        _commit_and_push(rule_id, implemented.function_name)
        implemented_rule_id = rule_id
        logger.info("Implemented rule %s at %s", rule_id, rule_path)
    except _ImplementationFailed:
        logger.exception("Rule implementation failed for idea '%s'", idea.title)

    # Write last_implemented.json for the new rule
    if implemented_rule_id:
        cycle_id = _cycle_id(config)
        if cycle_id is None:
            logger.warning(
                "No quote data available; skipping last_implemented.json write for %s",
                implemented_rule_id,
            )
        else:
            _write_last_implemented(state_dir, implemented_rule_id, cycle_id)

    # Write next_cycle_plan.json based on previous rule's performance
    _write_next_cycle_plan(state_dir, last_rule_id, analysis, config)


# ── Context loading ───────────────────────────────────────────────────────────


def _cycle_id(config: AppConfig) -> str | None:
    """Return a cycle identifier derived from the most recently processed quote."""
    latest = storage.latest_quote_time(config)
    return latest.strftime("%Y-%m-%dT%H-%M-%S") if latest else None


def _load_current_plan(state_dir: Path) -> NextCyclePlan:
    path = paths.next_cycle_plan(state_dir)
    if not path.exists():
        return NextCyclePlan(action="new_rule", description=None)
    try:
        return NextCyclePlan.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not parse next_cycle_plan.json; treating as new_rule")
        return NextCyclePlan(action="new_rule", description=None)


def _load_last_implemented(state_dir: Path) -> tuple[str | None, str | None]:
    path = paths.last_implemented(state_dir)
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("rule_id"), data.get("cycle_id")
    except Exception:
        logger.warning("Could not read last_implemented.json", exc_info=True)
        return None, None


def _get_rule_score(state_dir: Path, rule_id: str) -> RuleScore | None:
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


def _retrieve_top_k_traces(
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


def _run_relation_analysis(
    state_dir: Path, traces: list[EpisodicTrace], config: AppConfig
) -> _RelationAnalysis:
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
        output_type=_RelationAnalysis,
    )


# ── Idea generation ───────────────────────────────────────────────────────────


def _generate_idea(
    analysis: _RelationAnalysis,
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


# ── Next-cycle plan ───────────────────────────────────────────────────────────


def _write_next_cycle_plan(
    state_dir: Path,
    last_rule_id: str | None,
    analysis: _RelationAnalysis,
    config: AppConfig,
) -> None:
    plan_path = paths.next_cycle_plan(state_dir)

    if last_rule_id is None:
        # First cycle: no previous rule to evaluate
        plan = NextCyclePlan(action="continue", description=None)
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        logger.info("next_cycle_plan.json written: action=continue (first cycle)")
        return

    rule_score = _get_rule_score(state_dir, last_rule_id)
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


def _diagnose_failure(
    rule_score: RuleScore, analysis: _RelationAnalysis, config: AppConfig
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


def _write_last_implemented(state_dir: Path, rule_id: str, cycle_id: str) -> None:
    path = paths.last_implemented(state_dir)
    path.write_text(
        json.dumps({"rule_id": rule_id, "cycle_id": cycle_id}, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "last_implemented.json written: rule_id=%s cycle_id=%s", rule_id, cycle_id
    )


# ── Code generation ───────────────────────────────────────────────────────────


def _next_rule_path(idea: RuleIdea) -> tuple[str, Path]:
    """Return (versioned_rule_id, path_to_new_file)."""
    if idea.kind == "modify_rule" and idea.target_rule:
        base = re.sub(r"_v\d+$", "", idea.target_rule)
        folder = _RULES_DIR / base
        existing = sorted(folder.glob("v*.py")) if folder.exists() else []
        if existing:
            m = re.match(r"v(\d+)\.py$", existing[-1].name)
            next_ver = (int(m.group(1)) + 1) if m else 2
        else:
            next_ver = 1
        rule_id = f"{base}_v{next_ver}"
        return rule_id, folder / f"v{next_ver}.py"
    else:
        existing_nums = [
            int(m.group(1))
            for d in _RULES_DIR.iterdir()
            if d.is_dir() and (m := re.match(r"rule_(\d+)_", d.name))
        ]
        next_num = (max(existing_nums) + 1) if existing_nums else 1
        slug = re.sub(r"[^a-z0-9]+", "_", idea.title.lower()).strip("_")[:30]
        folder_name = f"rule_{next_num:02d}_{slug}"
        rule_id = f"{folder_name}_v1"
        return rule_id, _RULES_DIR / folder_name / "v1.py"


def _check_syntax(code: str) -> str | None:
    """Return an error description, or None if the code is a valid complete rule."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}"
    signal_fn = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "signal"
        ),
        None,
    )
    if signal_fn is None:
        return "Missing required top-level function: signal(data: MarketData)"
    has_return = any(isinstance(node, ast.Return) for node in ast.walk(signal_fn))
    if not has_return:
        return "signal() has no return statement — function body is likely incomplete"
    return None


def _load_target_source(target_rule: str) -> str | None:
    """Return the source of the rule being modified, or None if unavailable."""
    m = re.match(r"^(.+)_(v\d+)$", target_rule)
    if not m:
        return None
    path = _RULES_DIR / m.group(1) / f"{m.group(2)}.py"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _initial_code(idea: RuleIdea, llm: BaseChatModel) -> str:
    """First-pass code generation via plain-text LLM call."""
    existing_source = (
        _load_target_source(idea.target_rule)
        if idea.kind == "modify_rule" and idea.target_rule
        else None
    )
    existing_section = (
        f"Existing implementation to improve upon:\n{existing_source}\n\n"
        if existing_source
        else ""
    )
    user_prompt = (
        f"Implement this trading rule idea as a Python module.\n\n"
        f"Idea:\n{idea.model_dump_json(indent=2)}\n\n"
        f"{existing_section}"
        f"Available data models:\n{_MODELS_SOURCE}\n\n"
        f"Reference rule structure to follow exactly:\n{_REFERENCE_RULE}\n\n"
        "Requirements:\n"
        "1. The public entry-point function MUST be named exactly `signal` "
        "   with signature: def signal(data: MarketData) -> list[BuySignal | SellSignal].\n"
        "2. Available external packages: numpy, tensorflow, keras.\n"
        "3. Handle insufficient data gracefully (return [])."
    )
    response = llm.invoke(
        [SystemMessage(content=_IMPLEMENT_SYSTEM), HumanMessage(content=user_prompt)]
    )
    raw = response.content
    code = (
        raw
        if isinstance(raw, str)
        else "".join(p if isinstance(p, str) else p.get("text", "") for p in raw)
    ).strip()
    m = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
    if m:
        code = m.group(1).strip()
    return code


def _fix_with_diff(code: str, idea: RuleIdea, llm: BaseChatModel) -> str:
    """Ask the LLM for a diff to complete/fix `code`, then apply it."""
    structured = llm.with_structured_output(CodeDiff)
    user_prompt = (
        f"Idea:\n{idea.model_dump_json(indent=2)}\n\n"
        f"Available data models:\n{_MODELS_SOURCE}\n\n"
        f"Reference rule structure:\n{_REFERENCE_RULE}\n\n"
        "Entry-point function name: signal\n\n"
        f"Partial implementation to complete:\n{code}"
    )
    try:
        result = structured.invoke(
            [SystemMessage(content=_FIX_SYSTEM), HumanMessage(content=user_prompt)]
        )
        if not isinstance(result, CodeDiff):
            raise TypeError(f"Expected CodeDiff, got {type(result).__name__}")
        logger.debug("Applying %d change(s) from diff", len(result.changes))
        return apply_changes(code, result.changes)
    except Exception:
        logger.warning(
            "Diff generation/application failed; code unchanged", exc_info=True
        )
        return code


def _generate_code(idea: RuleIdea, rule_id: str, model: str) -> ImplementedRule:
    llm = make_llm(model)

    logger.info("Generating initial code for idea '%s' (%s)", idea.title, idea.idea_id)
    code = _initial_code(idea, llm)
    logger.debug("Initial code (%d chars):\n%s", len(code), code)

    for attempt in range(_MAX_FIX_ATTEMPTS):
        error = _check_syntax(code)
        if error is None:
            logger.info("Code passed validation after %d fix attempt(s)", attempt)
            break
        logger.warning(
            "Validation error (attempt %d/%d): %s\nCode:\n%s",
            attempt + 1,
            _MAX_FIX_ATTEMPTS,
            error,
            code,
        )
        code = _fix_with_diff(code, idea, llm)
        logger.debug(
            "Code after fix attempt %d (%d chars):\n%s", attempt + 1, len(code), code
        )
    else:
        error = _check_syntax(code)
        if error is not None:
            logger.error(
                "Code still failing after %d fix attempts: %s\nCode:\n%s",
                _MAX_FIX_ATTEMPTS,
                error,
                code,
            )
            raise _ImplementationFailed(
                f"Code still failing after {_MAX_FIX_ATTEMPTS} fix attempts: {error}"
            )

    return ImplementedRule(
        idea_id=idea.idea_id,
        rule_id=rule_id,
        function_name="signal",
        code=code,
    )


def _commit_and_push(rule_id: str, function_name: str) -> None:
    commit_msg = f"agent: implement rule {rule_id} ({function_name})"
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", "--set-upstream", "origin", "HEAD"], check=True)
        logger.info("Committed and pushed: %s", commit_msg)
    except subprocess.CalledProcessError:
        logger.exception("git commit/push failed after implementing %s", rule_id)
