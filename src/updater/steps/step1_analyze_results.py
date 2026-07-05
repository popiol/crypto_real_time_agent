"""Step 1 — Analyze results.

Groups evaluated buy-signal outcomes by rule_id and asks the LLM for a
qualitative interpretation of each rule's signals. Writes signal_evaluation.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent import storage
from src.agent.models import AppConfig
from src.updater.llm import llm_structured
from src.updater.models import RuleSignalEvaluation, SignalEvaluation

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a trading signal analyst. "
    "Given a set of buy-signal outcomes for a single trading rule, "
    "provide a concise qualitative interpretation: what patterns do you see, "
    "which pairs perform better or worse, and whether the rule shows genuine edge."
)


def run(config: AppConfig, state_dir: Path) -> None:
    evaluated = [
        r for r in storage.read_signals(config)
        if r.get("outcome") is not None and r.get("direction", "buy") == "buy"
    ]
    if not evaluated:
        logger.info("No evaluated signals yet; skipping step 1")
        return

    by_rule: dict[str, list[dict]] = {}
    for s in evaluated:
        by_rule.setdefault(s["rule_id"], []).append(s)

    rule_evals: list[RuleSignalEvaluation] = []
    for rule_id, signals in by_rule.items():
        try:
            result = llm_structured(
                model=config.llm_model,
                system=_SYSTEM,
                user=(
                    f"Rule: {rule_id}\n"
                    f"Signal outcomes ({len(signals)} total):\n"
                    f"{json.dumps(signals, indent=2)}"
                ),
                output_type=RuleSignalEvaluation,
            )
            rule_evals.append(result)
        except Exception:
            logger.warning("LLM evaluation failed for %s", rule_id, exc_info=True)
            rule_evals.append(RuleSignalEvaluation(
                rule_id=rule_id,
                signal_count=len(signals),
                notes="Evaluation unavailable.",
            ))

    out = state_dir / "signal_evaluation.json"
    out.write_text(SignalEvaluation(rules=rule_evals).model_dump_json(indent=2), encoding="utf-8")
    logger.info("signal_evaluation.json written (%d rules)", len(rule_evals))
