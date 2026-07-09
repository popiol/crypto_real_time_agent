"""Step 3 — Compare rule versions.

Groups registered rule versions by their base name and marks inferior versions
for dropping. Pure computation; no LLM call.

Writes version_comparison.json.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.agent.models import AppConfig
from src.updater.models import (
    RuleEvaluation,
    RuleVersionComparison,
    VersionComparisonResult,
)

logger = logging.getLogger(__name__)

_VERSION_SUFFIX = re.compile(r"_v(\d+)$")
_DROP_MARGIN = 0.15  # drop a version if its score is this much below the best


def run(config: AppConfig, state_dir: Path) -> None:
    rule_eval_path = state_dir / "rule_evaluation.json"
    if not rule_eval_path.exists():
        logger.info("rule_evaluation.json not found; skipping step 3")
        return

    evaluation = RuleEvaluation.model_validate_json(
        rule_eval_path.read_text(encoding="utf-8")
    )

    # Group all rules by base name (include candidates so multi-version rules are visible)
    by_base: dict[str, list] = {}
    for rule in evaluation.rules:
        base = _base_name(rule.rule_id)
        by_base.setdefault(base, []).append(rule)

    comparisons: list[RuleVersionComparison] = []
    for base, versions in by_base.items():
        if len(versions) < 2:
            continue
        scored = [v for v in versions if v.status != "candidate"]
        if len(scored) < 2:
            # Not enough evaluated versions to make a drop decision yet
            comparisons.append(
                RuleVersionComparison(
                    rule_name=base,
                    versions_compared=[v.rule_id for v in versions],
                    best_version=scored[0].rule_id if scored else versions[0].rule_id,
                    versions_to_drop=[],
                    rationale="Insufficient evaluated versions to compare; awaiting signal data.",
                )
            )
            continue
        best = max(scored, key=lambda r: r.score)
        to_drop = [
            v.rule_id
            for v in versions
            if v.rule_id != best.rule_id and best.score - v.score >= _DROP_MARGIN
        ]
        comparisons.append(
            RuleVersionComparison(
                rule_name=base,
                versions_compared=[v.rule_id for v in versions],
                best_version=best.rule_id,
                versions_to_drop=to_drop,
                rationale=(
                    f"Best version '{best.rule_id}' scores {best.score:.3f}"
                    + (
                        "; "
                        + "; ".join(
                            f"'{r.rule_id}' scores {r.score:.3f} (delta={best.score - r.score:.3f})"
                            for r in scored
                            if r.rule_id in to_drop
                        )
                        if to_drop
                        else ""
                    )
                ),
            )
        )

    multi_version_count = sum(1 for v in by_base.values() if len(v) >= 2)
    total_dropped = sum(len(c.versions_to_drop) for c in comparisons)
    result = VersionComparisonResult(
        comparisons=comparisons,
        summary=(
            f"Found {multi_version_count} multi-version rule(s); "
            f"marked {total_dropped} version(s) for dropping."
        ),
    )
    (state_dir / "version_comparison.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info(
        "version_comparison.json written (%d comparison(s), %d drop(s))",
        len(comparisons),
        total_dropped,
    )


def _base_name(rule_id: str) -> str:
    return _VERSION_SUFFIX.sub("", rule_id)
