"""Step 4 — Derive conclusions.

For each rule that has at least one version marked for dropping and at least
one remaining version, makes a separate LLM call to analyse what direction
the dropped modification took and why it underperformed, then proposes a
different direction to try.

Writes conclusions.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent.models import AppConfig
from src.updater.llm import llm_structured
from src.updater.models import (
    Conclusions,
    RuleEvaluation,
    RuleScore,
    VersionComparisonResult,
    VersionDirectionConclusion,
)

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a quantitative trading strategy analyst. "
    "You will be given performance data for multiple versions of the same trading rule. "
    "One or more versions underperformed and are being dropped. "
    "Your job is to identify what direction the failed modification took and why it likely "
    "did not improve performance, then propose a different direction to try next."
)


def run(config: AppConfig, state_dir: Path) -> None:
    rule_eval_path = state_dir / "rule_evaluation.json"
    version_cmp_path = state_dir / "version_comparison.json"

    if not rule_eval_path.exists() or not version_cmp_path.exists():
        logger.info("rule_evaluation.json or version_comparison.json not found; skipping step 4")
        return

    rule_eval = RuleEvaluation.model_validate_json(
        rule_eval_path.read_text(encoding="utf-8")
    )
    version_cmp = VersionComparisonResult.model_validate_json(
        version_cmp_path.read_text(encoding="utf-8")
    )

    scores_by_id: dict[str, RuleScore] = {r.rule_id: r for r in rule_eval.rules}

    conclusions: list[VersionDirectionConclusion] = []
    for cmp in version_cmp.comparisons:
        if not cmp.versions_to_drop:
            continue
        remaining = [v for v in cmp.versions_compared if v not in cmp.versions_to_drop]
        if not remaining:
            continue

        versions_data = [
            _rule_summary(scores_by_id[v])
            for v in cmp.versions_compared
            if v in scores_by_id
        ]
        if not versions_data:
            logger.warning("No evaluation data for %s versions; skipping", cmp.rule_name)
            continue

        user_prompt = (
            f"Rule: {cmp.rule_name}\n\n"
            f"Version performance data:\n{json.dumps(versions_data, indent=2)}\n\n"
            f"Versions being dropped (underperformed): {cmp.versions_to_drop}\n"
            f"Versions being kept: {remaining}\n\n"
            "Analyse what direction the dropped version(s) took compared to the kept version(s). "
            "Explain concisely why it likely failed. "
            "Then propose a clearly different modification direction that might improve performance."
        )

        try:
            result = llm_structured(
                model=config.llm_model,
                system=_SYSTEM,
                user=user_prompt,
                output_type=VersionDirectionConclusion,
            )
            result.rule_name = cmp.rule_name
            result.dropped_versions = cmp.versions_to_drop
            conclusions.append(result)
            logger.info(
                "Conclusion for %s: drop %s — %s",
                cmp.rule_name,
                cmp.versions_to_drop,
                result.failed_direction[:80],
            )
        except Exception:
            logger.warning("LLM call failed for rule %s", cmp.rule_name, exc_info=True)

    output = Conclusions(conclusions=conclusions)
    (state_dir / "conclusions.json").write_text(
        output.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("conclusions.json written (%d conclusion(s))", len(conclusions))


def _rule_summary(rule: RuleScore) -> dict:
    return {
        "rule_id": rule.rule_id,
        "description": rule.description,
        "score": rule.score,
        "signal_count": rule.signal_count,
        "avg_gain_pct": rule.avg_gain_pct,
        "positive_rate": rule.positive_rate,
        "status": rule.status,
    }
