"""Step 5 — Update long-term plan.

Revises the strategic direction based on the latest conclusions.
Writes long_term_plan.json.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from src.agent.models import AppConfig
from src.updater.llm import llm_structured
from src.updater.models import Conclusions, LongTermPlan, RuleEvaluation

logger = logging.getLogger(__name__)


def run(config: AppConfig, state_dir: Path) -> None:
    conclusions_path = state_dir / "conclusions.json"
    if not conclusions_path.exists():
        logger.info("conclusions.json not found; skipping step 5")
        return

    conclusions = Conclusions.model_validate_json(
        conclusions_path.read_text(encoding="utf-8")
    )

    rule_eval_text = "No rule evaluation available."
    rule_eval_path = state_dir / "rule_evaluation.json"
    if rule_eval_path.exists():
        try:
            rule_eval = RuleEvaluation.model_validate_json(
                rule_eval_path.read_text(encoding="utf-8")
            )
            rule_eval_text = rule_eval.model_dump_json(indent=2)
        except Exception:
            pass

    existing_plan_text = "No existing plan."
    plan_path = state_dir / "long_term_plan.json"
    if plan_path.exists():
        existing_plan_text = plan_path.read_text(encoding="utf-8")

    now_iso = datetime.now(timezone.utc).isoformat()

    result = llm_structured(
        model=config.llm_model,
        system=(
            "You are a trading strategy director for an autonomous crypto trading agent. "
            "Maintain a clear, actionable long-term plan that guides rule development."
        ),
        user=(
            "Revise the long-term strategy plan based on the current rule performance and version conclusions.\n\n"
            f"Rule evaluation:\n{rule_eval_text}\n\n"
            f"Version conclusions (directions that failed and proposed alternatives):\n{conclusions.model_dump_json(indent=2)}\n\n"
            f"Current plan:\n{existing_plan_text}\n\n"
            f"Set updated_at to: {now_iso}"
        ),
        output_type=LongTermPlan,
    )

    plan_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info("long_term_plan.json written")
