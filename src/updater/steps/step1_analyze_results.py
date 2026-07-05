"""Step 1 — Analyze results.

Groups evaluated buy-signal outcomes by rule_id and computes per-rule
metrics numerically. No LLM call — pure computation from the signal ledger.

Writes signal_evaluation.json.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from src.agent import storage
from src.agent.models import AppConfig
from src.updater.models import PairMetrics, RuleSignalEvaluation, SignalEvaluation

logger = logging.getLogger(__name__)


def run(config: AppConfig, state_dir: Path) -> None:
    evaluated = [
        r for r in storage.read_signals(config)
        if r.get("outcome") is not None and r.get("direction", "buy") == "buy"
    ]
    if not evaluated:
        logger.info("No evaluated signals yet; skipping step 1")
        return

    by_rule: dict[str, list[dict]] = defaultdict(list)
    for s in evaluated:
        by_rule[s["rule_id"]].append(s)

    rule_evals: list[RuleSignalEvaluation] = []
    for rule_id, signals in by_rule.items():
        rule_evals.append(_compute(rule_id, signals))

    out = state_dir / "signal_evaluation.json"
    out.write_text(SignalEvaluation(rules=rule_evals).model_dump_json(indent=2), encoding="utf-8")
    logger.info("signal_evaluation.json written (%d rules)", len(rule_evals))


def _compute(rule_id: str, signals: list[dict]) -> RuleSignalEvaluation:
    gains = [s["outcome"]["gain_pct"] for s in signals]
    positive = sum(1 for g in gains if g > 0)

    by_exit: dict[str, int] = defaultdict(int)
    for s in signals:
        by_exit[s["outcome"].get("exit_reason", "unknown")] += 1

    by_pair: dict[str, list[float]] = defaultdict(list)
    for s in signals:
        by_pair[s["pair"]].append(s["outcome"]["gain_pct"])

    pair_metrics = [
        PairMetrics(
            pair=pair,
            signal_count=len(pair_gains),
            avg_gain_pct=sum(pair_gains) / len(pair_gains),
            positive_rate=sum(1 for g in pair_gains if g > 0) / len(pair_gains),
        )
        for pair, pair_gains in sorted(by_pair.items())
    ]

    return RuleSignalEvaluation(
        rule_id=rule_id,
        signal_count=len(signals),
        positive_rate=positive / len(signals),
        avg_gain_pct=sum(gains) / len(gains),
        by_exit_reason=dict(by_exit),
        by_pair=pair_metrics,
    )
