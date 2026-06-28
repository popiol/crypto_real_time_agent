"""Step 2 — Analyze current rules.

For each registered rule:
  - Generates a plain-language description via LLM (once per version; cached).
  - Computes numeric scores from the signal ledger.

Writes rule_descriptions.json and rule_evaluation.json.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
from pathlib import Path

from pydantic import BaseModel

from src.agent.models import AppConfig
from src.strategy.strategy import ACTIVE_RULES
from src.updater.llm import llm_structured
from src.updater.models import (
    RuleDescription,
    RuleDescriptions,
    RuleEvaluation,
    RuleScore,
)

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
    ledger_signals = _read_ledger(Path(config.data_dir) / "signals.ndjson")

    desc_path = state_dir / "rule_descriptions.json"
    desc_cache: dict[str, str] = _load_desc_cache(desc_path)

    scores: list[RuleScore] = []
    for rule_fn in ACTIVE_RULES:
        module_rule_id = rule_fn.__module__.split(".")[-1]
        signal_rule_id = _get_signal_rule_id(rule_fn)
        description = _describe(module_rule_id, rule_fn, desc_cache, config.llm_model)
        desc_cache[module_rule_id] = description
        scores.append(_score(module_rule_id, signal_rule_id, description, ledger_signals, config))

    # Persist updated description cache
    desc_path.write_text(
        RuleDescriptions(
            rules=[RuleDescription(rule_id=k, description=v) for k, v in desc_cache.items()]
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    # Generate summary and write rule evaluation
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


def _load_desc_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        parsed = RuleDescriptions.model_validate_json(path.read_text(encoding="utf-8"))
        return {r.rule_id: r.description for r in parsed.rules}
    except Exception:
        logger.warning("Could not parse rule_descriptions.json; rebuilding")
        return {}


def _get_signal_rule_id(rule_fn) -> str:
    try:
        module = importlib.import_module(rule_fn.__module__)
        return getattr(module, "RULE_ID", rule_fn.__name__)
    except Exception:
        return rule_fn.__name__


def _describe(module_rule_id: str, rule_fn, cache: dict[str, str], model: str) -> str:
    if module_rule_id in cache:
        return cache[module_rule_id]
    try:
        source = inspect.getsource(rule_fn)
    except Exception:
        source = f"# source unavailable for {module_rule_id}"
    try:
        result = llm_structured(
            model=model,
            system=_DESCRIBE_SYSTEM,
            user=f"Rule ID: {module_rule_id}\n\nSource:\n{source}",
            output_type=_RuleDesc,
        )
        return result.description
    except Exception:
        logger.warning("LLM description failed for %s", module_rule_id, exc_info=True)
        return f"No description available for {module_rule_id}."


def _score(
    module_rule_id: str,
    signal_rule_id: str,
    description: str,
    ledger_signals: list[dict],
    config: AppConfig,
) -> RuleScore:
    matching = [
        s for s in ledger_signals
        if s.get("rule_id") == signal_rule_id and s.get("outcome") is not None
    ]
    signal_count = len(matching)

    if signal_count == 0:
        return RuleScore(
            rule_id=module_rule_id,
            description=description,
            signal_count=0,
            avg_gain_24h=0.0,
            max_gain_24h=0.0,
            positive_rate=0.0,
            score=0.0,
            status="candidate",
        )

    gains = [s["outcome"]["gain_24h_pct"] for s in matching]
    max_gains = [s["outcome"]["max_gain_24h_pct"] for s in matching]
    avg_gain = sum(gains) / len(gains)
    max_gain = max(max_gains)
    positive_rate = sum(1 for g in gains if g > 0) / len(gains)
    composite = round(0.6 * positive_rate + 0.4 * min(1.0, max(0.0, avg_gain / 5.0)), 4)

    if signal_count < config.rule_min_signals:
        status = "candidate"
    elif composite >= config.rule_deprecation_threshold:
        status = "active"
    else:
        status = "deprecate"

    return RuleScore(
        rule_id=module_rule_id,
        description=description,
        signal_count=signal_count,
        avg_gain_24h=avg_gain,
        max_gain_24h=max_gain,
        positive_rate=positive_rate,
        score=composite,
        status=status,
    )


def _read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
