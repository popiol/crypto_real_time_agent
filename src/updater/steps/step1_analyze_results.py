"""Step 1 — Analyze results.

Reads evaluated signals from the ledger, sends them to the LLM, and writes
signal_evaluation.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent.models import AppConfig
from src.updater.llm import llm_structured
from src.updater.models import SignalEvaluation

logger = logging.getLogger(__name__)


def run(config: AppConfig, state_dir: Path) -> None:
    ledger = Path(config.data_dir) / "signals.ndjson"
    if not ledger.exists():
        logger.info("No signal ledger found; skipping step 1")
        return

    evaluated = [
        r for r in _read_ledger(ledger)
        if r.get("outcome") is not None
    ]
    if not evaluated:
        logger.info("No evaluated signals yet; skipping step 1")
        return

    user = (
        "Analyze these buy-signal outcomes from an automated crypto trading system "
        "and return a structured evaluation.\n\n"
        f"Signal outcomes (JSON):\n{json.dumps(evaluated, indent=2)}"
    )
    result = llm_structured(
        model=config.llm_model,
        system=(
            "You are a trading signal analyst. "
            "Evaluate the performance of automated buy signals based on 24-hour price outcomes."
        ),
        user=user,
        output_type=SignalEvaluation,
    )

    out = state_dir / "signal_evaluation.json"
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info("signal_evaluation.json written (%d signals)", len(evaluated))


def _read_ledger(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed ledger line %d", lineno)
    return records
