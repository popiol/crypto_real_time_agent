"""Signal outcome evaluator.

Runs every hour. Evaluates pending buy signals using two triggers:

  1. Sell signal match — if a sell signal was emitted for the same pair after
     this buy signal, outcome = sell_price / buy_price - 1.

  2. Timeout — if the buy signal is older than TIMEOUT_DAYS with no matching
     sell signal, outcome = current_price / buy_price - 1 (using latest warm candle).

Sell signals are not evaluated; they serve only as triggers for buy signal evaluation.
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

_TIMEOUT = timedelta(days=20)


def evaluate_pending_signals(config: AppConfig) -> None:
    ledger = Path(config.data_dir) / "signals.ndjson"
    if not ledger.exists():
        return

    now = datetime.now(timezone.utc)
    raw_records = _load_ledger(ledger)

    sell_index = _build_sell_index(raw_records)

    lines: list[str] = []
    updated = False

    for record, original_line in raw_records:
        if record is None:
            lines.append(original_line)
            continue

        if record.get("outcome") is not None:
            lines.append(original_line)
            continue

        if record.get("direction", "buy") != "buy":
            lines.append(original_line)
            continue

        pair = record.get("pair", "")
        price_at_signal = record.get("price_at_signal")
        emitted_at = _parse_dt(record.get("emitted_at", ""))

        if not pair or price_at_signal is None or emitted_at is None:
            lines.append(original_line)
            continue

        outcome = _resolve_outcome(
            pair, emitted_at, float(price_at_signal), now, sell_index, config
        )
        if outcome is None:
            lines.append(original_line)
            continue

        record["outcome"] = outcome
        lines.append(json.dumps(record))
        updated = True

    if updated:
        _rewrite_ledger(ledger, lines)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_ledger(path: Path) -> list[tuple[dict | None, str]]:
    result: list[tuple[dict | None, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            result.append((json.loads(line), line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed ledger line %d", lineno)
            result.append((None, line))
    return result


def _build_sell_index(
    raw_records: list[tuple[dict | None, str]],
) -> dict[str, list[tuple[datetime, float]]]:
    """Map pair → list of (emitted_at, price) for all sell signals."""
    index: dict[str, list[tuple[datetime, float]]] = {}
    for record, _ in raw_records:
        if record is None or record.get("direction") != "sell":
            continue
        pair = record.get("pair", "")
        emitted_at = _parse_dt(record.get("emitted_at", ""))
        price = record.get("price_at_signal")
        if pair and emitted_at is not None and price is not None:
            index.setdefault(pair, []).append((emitted_at, float(price)))
    return index


def _resolve_outcome(
    pair: str,
    emitted_at: datetime,
    price_at_signal: float,
    now: datetime,
    sell_index: dict[str, list[tuple[datetime, float]]],
    config: AppConfig,
) -> dict | None:
    sells_after = [
        (dt, p) for dt, p in sell_index.get(pair, []) if dt > emitted_at
    ]
    if sells_after:
        _, exit_price = min(sells_after, key=lambda x: x[0])
        return {
            "evaluated_at": now.isoformat(),
            "exit_price": exit_price,
            "exit_reason": "sell_signal",
            "gain_pct": exit_price / price_at_signal - 1,
        }

    if now - emitted_at >= _TIMEOUT:
        exit_price = _latest_price(pair, config)
        if exit_price is None:
            logger.warning("No price data for %s — cannot evaluate timed-out signal", pair)
            return None
        return {
            "evaluated_at": now.isoformat(),
            "exit_price": exit_price,
            "exit_reason": "timeout",
            "gain_pct": exit_price / price_at_signal - 1,
        }

    return None


def _latest_price(pair: str, config: AppConfig) -> float | None:
    warm = storage.read_warm_candles(pair, config)
    return warm[-1].close if warm else None


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
