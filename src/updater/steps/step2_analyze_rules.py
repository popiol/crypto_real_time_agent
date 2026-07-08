"""Step 2 — Analyze current rules.

For each registered rule:
  - Generates a plain-language description via LLM (once per version; cached
    from the previous run's rule_evaluation.json).
  - Computes numeric scores from the signal ledger.

Writes rule_evaluation.json (sole source of descriptions + scores).
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path

from pydantic import BaseModel

import importlib
import sys

from src.agent import storage
from src.agent.models import AppConfig
from src.updater.llm import llm_structured
from src.updater.models import RuleEvaluation, RuleScore

logger = logging.getLogger(__name__)

_DESCRIBE_SYSTEM = (
    "You are a trading systems analyst. "
    "Describe the given trading rule concisely in 2-3 sentences covering: "
    "what market condition it detects, what signal it emits, and its underlying hypothesis."
)


class _EvalSummary(BaseModel):
    summary: str


class _RuleDesc(BaseModel):
    description: str


def run(config: AppConfig, state_dir: Path) -> None:
    strategy = sys.modules.get("src.strategy.strategy")
    if strategy is not None:
        importlib.reload(strategy)
    from src.strategy.strategy import ACTIVE_RULES

    ledger_signals = storage.read_signals(config)

    # Load caches from the prior run's rule_evaluation.json
    prior_eval_path = state_dir / "rule_evaluation.json"
    desc_cache: dict[str, str] = _load_desc_cache(prior_eval_path)
    zero_cycles_cache: dict[str, int] = _load_zero_cycles_cache(prior_eval_path)

    scores: list[RuleScore] = []
    for rule_module in ACTIVE_RULES:
        parts = rule_module.__name__.split(".")
        rule_id = f"{parts[-2]}_{parts[-1]}"  # e.g. rule_01_spread_compression_v1
        description = _describe(rule_id, rule_module, desc_cache, config.llm_model)
        desc_cache[rule_id] = description
        scores.append(_score(rule_id, description, ledger_signals, zero_cycles_cache, config))

    try:
        summary_result = llm_structured(
            model=config.llm_model,
            system="You are a trading strategy analyst.",
            user=(
                "Summarise the overall health of this rule set in one paragraph.\n\n"
                f"Rules:\n{json.dumps([s.model_dump() for s in scores], indent=2)}"
            ),
            output_type=_EvalSummary,
        )
        summary = summary_result.summary
    except Exception:
        logger.warning("Rule evaluation summary LLM call failed", exc_info=True)
        summary = ""

    (state_dir / "rule_evaluation.json").write_text(
        RuleEvaluation(rules=scores, summary=summary).model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info("rule_evaluation.json written (%d rules)", len(scores))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_desc_cache(rule_eval_path: Path) -> dict[str, str]:
    """Extract description cache from the previous rule_evaluation.json."""
    if not rule_eval_path.exists():
        return {}
    try:
        prior = RuleEvaluation.model_validate_json(rule_eval_path.read_text(encoding="utf-8"))
        return {r.rule_id: r.description for r in prior.rules if r.description}
    except Exception:
        logger.warning("Could not parse prior rule_evaluation.json for description cache")
        return {}


def _load_zero_cycles_cache(rule_eval_path: Path) -> dict[str, int]:
    """Extract consecutive zero-signal cycle counts from the previous rule_evaluation.json."""
    if not rule_eval_path.exists():
        return {}
    try:
        prior = RuleEvaluation.model_validate_json(rule_eval_path.read_text(encoding="utf-8"))
        return {r.rule_id: r.zero_signal_cycles for r in prior.rules}
    except Exception:
        return {}


def _describe(rule_id: str, rule_module, cache: dict[str, str], model: str) -> str:
    if rule_id in cache:
        return cache[rule_id]
    try:
        source = inspect.getsource(rule_module.signal)
    except Exception:
        source = f"# source unavailable for {rule_id}"
    try:
        result = llm_structured(
            model=model,
            system=_DESCRIBE_SYSTEM,
            user=f"Rule ID: {rule_id}\n\nSource:\n{source}",
            output_type=_RuleDesc,
        )
        return result.description
    except Exception:
        logger.warning("LLM description failed for %s", rule_id, exc_info=True)
        return f"No description available for {rule_id}."


def _score(
    rule_id: str,
    description: str,
    ledger_signals: list[dict],
    zero_cycles_cache: dict[str, int],
    config: AppConfig,
) -> RuleScore:
    matching = [
        s for s in ledger_signals
        if s.get("rule_id") == rule_id and s.get("outcome") is not None
    ]
    signal_count = len(matching)

    if signal_count == 0:
        zero_signal_cycles = zero_cycles_cache.get(rule_id, 0) + 1
        status = "deprecate" if zero_signal_cycles >= config.rule_zero_signal_max_cycles else "candidate"
        if status == "deprecate":
            logger.warning(
                "Rule %s has emitted 0 signals for %d consecutive cycles; marking for deprecation",
                rule_id, zero_signal_cycles,
            )
        return RuleScore(
            rule_id=rule_id,
            description=description,
            signal_count=0,
            evaluation_days=0,
            avg_gain_pct=0.0,
            positive_rate=0.0,
            avg_gain_24h=0.0,
            max_gain_24h=0.0,
            score=0.0,
            status=status,
            zero_signal_cycles=zero_signal_cycles,
        )

    gains_pct = [s["outcome"]["gain_pct"] for s in matching]
    avg_gain_pct = sum(gains_pct) / len(gains_pct)
    positive_rate = sum(1 for g in gains_pct if g > 0) / len(gains_pct)

    with_24h = [s["outcome"] for s in matching if "gain_24h_pct" in s["outcome"]]
    avg_gain_24h = sum(o["gain_24h_pct"] for o in with_24h) / len(with_24h) if with_24h else 0.0
    max_gain_24h = max(o["max_gain_24h_pct"] for o in with_24h) if with_24h else 0.0

    # Evaluation span in days (from first to last evaluated signal)
    timestamps = [s["emitted_at"] for s in matching if s.get("emitted_at")]
    evaluation_days = 0
    if len(timestamps) >= 2:
        from datetime import datetime
        def _parse(ts) -> datetime:
            if isinstance(ts, datetime):
                return ts
            return datetime.fromisoformat(str(ts))
        evaluation_days = (_parse(max(timestamps)) - _parse(min(timestamps))).days

    # Score: avg_gain_pct normalised to [0,1] where 0 = -10%, 0.5 = 0%, 1.0 = +10%
    score = round(max(0.0, min(1.0, (avg_gain_pct + 0.10) / 0.20)), 4)

    # Deprecation: time-aware
    # - candidate  : not enough signals yet
    # - short eval : deprecate only on severe loss (< rule_early_deprecation_gain)
    # - mature eval : deprecate on zero or below-zero avg gain
    mature = evaluation_days >= config.rule_mature_days
    if signal_count < config.rule_min_signals:
        status = "candidate"
    elif mature and avg_gain_pct <= config.rule_mature_deprecation_gain:
        status = "deprecate"
    elif not mature and avg_gain_pct < config.rule_early_deprecation_gain:
        status = "deprecate"
    else:
        status = "active"

    return RuleScore(
        rule_id=rule_id,
        description=description,
        signal_count=signal_count,
        evaluation_days=evaluation_days,
        avg_gain_pct=avg_gain_pct,
        positive_rate=positive_rate,
        avg_gain_24h=avg_gain_24h,
        max_gain_24h=max_gain_24h,
        score=score,
        status=status,
    )
