"""Step 4 — Derive conclusions.

Interprets rule evaluations holistically via LLM.
Writes conclusions.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent.models import AppConfig
from src.updater.llm import llm_structured
from src.updater.models import Conclusions, RuleEvaluation, VersionComparisonResult

logger = logging.getLogger(__name__)


def run(config: AppConfig, state_dir: Path) -> None:
    rule_eval_path = state_dir / "rule_evaluation.json"
    if not rule_eval_path.exists():
        logger.info("rule_evaluation.json not found; skipping step 4")
        return

    rule_eval = RuleEvaluation.model_validate_json(
        rule_eval_path.read_text(encoding="utf-8")
    )

    rules_text = json.dumps([r.model_dump() for r in rule_eval.rules], indent=2)

    version_cmp_text = ""
    version_cmp_path = state_dir / "version_comparison.json"
    if version_cmp_path.exists():
        version_cmp = VersionComparisonResult.model_validate_json(
            version_cmp_path.read_text(encoding="utf-8")
        )
        version_cmp_text = f"\n\nVersion comparison:\n{version_cmp.model_dump_json(indent=2)}"

    result = llm_structured(
        model=config.llm_model,
        system=(
            "You are a quantitative trading strategy analyst. "
            "Interpret rule performance data and derive actionable strategic conclusions."
        ),
        user=(
            "Based on the following rule evaluations, derive a list of strategic conclusions. "
            "Each conclusion should reference the specific rule(s) it applies to.\n\n"
            f"Rule evaluations:\n{rules_text}"
            f"{version_cmp_text}"
        ),
        output_type=Conclusions,
    )

    (state_dir / "conclusions.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("conclusions.json written")
