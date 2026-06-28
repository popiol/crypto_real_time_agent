"""Signal outcome evaluator.

Runs every hour. For each signal in the ledger that is older than 24 hours
and still has outcome = null, reconstructs the price window from the warm
tier and fills in the outcome fields.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.agent import storage
from src.agent.models import AppConfig

logger = logging.getLogger(__name__)

_EVALUATION_WINDOW = timedelta(hours=24)


def evaluate_pending_signals(config: AppConfig) -> None:
    """Fill in outcomes for signals emitted more than 24 hours ago."""
    ledger = Path(config.data_dir) / "signals.ndjson"
    if not ledger.exists():
        return

    now = datetime.now(timezone.utc)
    cutoff = now - _EVALUATION_WINDOW

    lines: list[str] = []
    updated = False

    with ledger.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.rstrip("\n")
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed ledger line %d", lineno)
                lines.append(line)
                continue

            if record.get("outcome") is not None:
                lines.append(line)
                continue

            emitted_at = _parse_dt(record.get("emitted_at", ""))
            if emitted_at is None or emitted_at > cutoff:
                lines.append(line)
                continue

            pair = record.get("pair", "")
            price_at_signal = record.get("price_at_signal")
            if not pair or price_at_signal is None:
                lines.append(line)
                continue

            outcome = _compute_outcome(pair, emitted_at, float(price_at_signal), now, config)
            if outcome is None:
                lines.append(line)
                continue

            record["outcome"] = outcome
            lines.append(json.dumps(record))
            updated = True
            logger.info(
                "Evaluated signal %s (%s %s): gain_24h=%.2f%%",
                record.get("signal_id", "?"),
                record.get("rule_id", "?"),
                pair,
                outcome["gain_24h_pct"],
            )

    if updated:
        _rewrite_ledger(ledger, lines)


def _compute_outcome(
    pair: str,
    emitted_at: datetime,
    price_at_signal: float,
    now: datetime,
    config: AppConfig,
) -> dict | None:
    window_end = emitted_at + _EVALUATION_WINDOW
    warm = storage.read_warm_candles(pair, config)
    in_window = [c for c in warm if emitted_at <= c.hour <= window_end]

    if not in_window:
        logger.warning(
            "No warm candles for %s in window %s to %s — cannot evaluate yet",
            pair,
            emitted_at.isoformat(),
            window_end.isoformat(),
        )
        return None

    closest = min(in_window, key=lambda c: abs((c.hour - window_end).total_seconds()))
    price_24h = closest.close
    max_price_24h = max(c.high for c in in_window)

    return {
        "evaluated_at": now.isoformat(),
        "price_24h": price_24h,
        "max_price_24h": max_price_24h,
        "gain_24h_pct": (price_24h - price_at_signal) / price_at_signal * 100,
        "max_gain_24h_pct": (max_price_24h - price_at_signal) / price_at_signal * 100,
    }


def _parse_dt(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _rewrite_ledger(path: Path, lines: list[str]) -> None:
    tmp = path.with_suffix(".ndjson.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")
        os.replace(tmp, path)
    except Exception:
        logger.exception("Failed to rewrite signal ledger")
        tmp.unlink(missing_ok=True)
