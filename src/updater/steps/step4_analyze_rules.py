"""Step 2 — Analyze current rules.

For each registered rule:
  - Generates a plain-language description via LLM (once per version; cached
    from the previous run's rule_evaluation.json).
  - Computes numeric scores from the signal ledger.

Writes rule_evaluation.json (sole source of descriptions + scores).
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path

from pydantic import BaseModel

import importlib
import sys

from src.agent import storage
from src.agent.models import AppConfig
from src.updater.llm import llm_structured
from src.updater.models import GainByVolatility, RuleEvaluation, RuleScore

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
    strategy = sys.modules.get("src.strategy.strategy")
    if strategy is not None:
        importlib.reload(strategy)
    from src.strategy.strategy import ACTIVE_RULES

    ledger_signals = storage.read_signals(config)
    transaction_gains = _load_transaction_gains(config)

    # Load caches from the prior run's rule_evaluation.json
    prior_eval_path = state_dir / "rule_evaluation.json"
    desc_cache: dict[str, str] = _load_desc_cache(prior_eval_path)
    zero_cycles_cache: dict[str, int] = _load_zero_cycles_cache(prior_eval_path)

    scores: list[RuleScore] = []
    for rule_module in ACTIVE_RULES:
        parts = rule_module.__name__.split(".")
        rule_id = f"{parts[-2]}_{parts[-1]}"  # e.g. rule_01_spread_compression_v1
        description = _describe(rule_id, rule_module, desc_cache, config.llm_model)
        desc_cache[rule_id] = description
        scores.append(_score(rule_id, description, ledger_signals, zero_cycles_cache, transaction_gains, config))

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


def _load_desc_cache(rule_eval_path: Path) -> dict[str, str]:
    """Extract description cache from the previous rule_evaluation.json."""
    if not rule_eval_path.exists():
        return {}
    try:
        prior = RuleEvaluation.model_validate_json(rule_eval_path.read_text(encoding="utf-8"))
        return {r.rule_id: r.description for r in prior.rules if r.description}
    except Exception:
        logger.warning("Could not parse prior rule_evaluation.json for description cache")
        return {}


def _load_zero_cycles_cache(rule_eval_path: Path) -> dict[str, int]:
    """Extract consecutive zero-signal cycle counts from the previous rule_evaluation.json."""
    if not rule_eval_path.exists():
        return {}
    try:
        prior = RuleEvaluation.model_validate_json(rule_eval_path.read_text(encoding="utf-8"))
        return {r.rule_id: r.zero_signal_cycles for r in prior.rules}
    except Exception:
        return {}


def _describe(rule_id: str, rule_module, cache: dict[str, str], model: str) -> str:
    if rule_id in cache:
        return cache[rule_id]
    try:
        source = inspect.getsource(rule_module.signal)
    except Exception:
        source = f"# source unavailable for {rule_id}"
    try:
        result = llm_structured(
            model=model,
            system=_DESCRIBE_SYSTEM,
            user=f"Rule ID: {rule_id}\n\nSource:\n{source}",
            output_type=_RuleDesc,
        )
        return result.description
    except Exception:
        logger.warning("LLM description failed for %s", rule_id, exc_info=True)
        return f"No description available for {rule_id}."


def _load_transaction_gains(config: AppConfig) -> dict[str, float]:
    """Return average transaction gain_pct per rule_id from portfolio/transactions.json."""
    path = Path(config.data_dir) / "portfolio" / "transactions.json"
    if not path.exists():
        return {}
    try:
        txns = json.loads(path.read_text(encoding="utf-8"))
        by_rule: dict[str, list[float]] = {}
        for t in txns:
            rule_id = t.get("rule_id")
            gain = t.get("gain_pct")
            if rule_id and gain is not None:
                by_rule.setdefault(rule_id, []).append(gain)
        return {rule_id: sum(gains) / len(gains) for rule_id, gains in by_rule.items()}
    except Exception:
        logger.warning("Could not read transactions.json for avg_transaction_gain", exc_info=True)
        return {}


def _weekly_counts(signals: list[dict]) -> list[int]:
    """Return signal counts per calendar week (ISO), oldest week first."""
    from collections import defaultdict
    from datetime import datetime, timezone

    def _parse(ts) -> datetime:
        dt = datetime.fromisoformat(str(ts))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

    by_week: dict[tuple[int, int], int] = defaultdict(int)
    for s in signals:
        ts = s.get("emitted_at")
        if not ts:
            continue
        iso = _parse(ts).isocalendar()
        by_week[(iso.year, iso.week)] += 1

    if not by_week:
        return []
    sorted_weeks = sorted(by_week)
    return [by_week[w] for w in sorted_weeks]


def _signal_trend(weekly: list[int]) -> str:
    if len(weekly) < 3:
        return "stable"
    first3 = sum(weekly[:3]) / 3
    last3 = sum(weekly[-3:]) / 3
    if last3 > first3:
        return "increasing"
    if last3 < first3:
        return "decreasing"
    return "stable"


def _bbw(closes: list[float], period: int = 20) -> float | None:
    """Bollinger Band Width = 4 * std(closes[-period:]) / sma(closes[-period:])."""
    if len(closes) < period:
        return None
    window = closes[-period:]
    sma = sum(window) / period
    if sma == 0:
        return None
    variance = sum((c - sma) ** 2 for c in window) / period
    std = variance ** 0.5
    return 4 * std / sma


def _gain_by_volatility(signals: list[dict], config: AppConfig) -> GainByVolatility:
    """Bucket signal gains by Bollinger Band Width at signal time."""
    from datetime import datetime, timedelta, timezone

    def _parse(ts) -> datetime:
        dt = datetime.fromisoformat(str(ts))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

    # Gather signals with timestamps; fetch warm candles per pair
    bbw_gains: list[tuple[float, float]] = []  # (bbw, gain_pct)
    candle_cache: dict[str, list] = {}

    for s in signals:
        pair = s.get("pair")
        ts = s.get("emitted_at")
        gain = s.get("outcome", {}).get("gain_pct")
        if not pair or not ts or gain is None:
            continue
        signal_time = _parse(ts)
        since = signal_time - timedelta(hours=22)
        if pair not in candle_cache:
            candle_cache[pair] = storage.read_warm_candles_range(pair, since, config)
        candles = [c for c in candle_cache[pair] if c.hour <= signal_time]
        closes = [c.close for c in candles]
        bbw = _bbw(closes)
        if bbw is not None:
            bbw_gains.append((bbw, gain))

    if not bbw_gains:
        return GainByVolatility()

    bbws = sorted(bg[0] for bg in bbw_gains)
    n = len(bbws)
    low_threshold = bbws[n // 3]
    high_threshold = bbws[(2 * n) // 3]

    def _avg(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    low_gains = [g for bw, g in bbw_gains if bw <= low_threshold]
    mid_gains = [g for bw, g in bbw_gains if low_threshold < bw <= high_threshold]
    high_gains = [g for bw, g in bbw_gains if bw > high_threshold]

    return GainByVolatility(
        low=_avg(low_gains),
        medium=_avg(mid_gains),
        high=_avg(high_gains),
    )


def _percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile of values using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = p / 100 * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[lo]
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def _streaks(gains: list[float]) -> tuple[int, int]:
    """Return (longest_win_streak, longest_loss_streak)."""
    max_win = max_loss = cur_win = cur_loss = 0
    for g in gains:
        if g > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def _score(
    rule_id: str,
    description: str,
    ledger_signals: list[dict],
    zero_cycles_cache: dict[str, int],
    transaction_gains: dict[str, float],
    config: AppConfig,
) -> RuleScore:
    matching = [
        s for s in ledger_signals
        if s.get("rule_id") == rule_id and s.get("outcome") is not None
    ]
    signal_count = len(matching)

    if signal_count == 0:
        zero_signal_cycles = zero_cycles_cache.get(rule_id, 0) + 1
        status = "deprecate" if zero_signal_cycles >= config.rule_zero_signal_max_cycles else "candidate"
        if status == "deprecate":
            logger.warning(
                "Rule %s has emitted 0 signals for %d consecutive cycles; marking for deprecation",
                rule_id, zero_signal_cycles,
            )
        return RuleScore(
            rule_id=rule_id,
            description=description,
            signal_count=0,
            evaluation_days=0,
            avg_gain_pct=0.0,
            recent_avg_gain_pct=0.0,
            avg_transaction_gain=transaction_gains.get(rule_id, 0.0),
            positive_rate=0.0,
            avg_gain_24h=0.0,
            max_gain_24h=0.0,
            score=0.0,
            status=status,
            zero_signal_cycles=zero_signal_cycles,
        )

    from datetime import datetime, timedelta, timezone

    def _parse(ts) -> datetime:
        if isinstance(ts, datetime):
            return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
        dt = datetime.fromisoformat(str(ts))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

    gains_pct = [s["outcome"]["gain_pct"] for s in matching]
    avg_gain_pct = sum(gains_pct) / len(gains_pct)
    positive_rate = sum(1 for g in gains_pct if g > 0) / len(gains_pct)
    min_gain_pct = min(gains_pct)
    p25_gain_pct = _percentile(gains_pct, 25)
    p75_gain_pct = _percentile(gains_pct, 75)
    longest_win_streak, longest_loss_streak = _streaks(gains_pct)
    weekly = _weekly_counts(matching)
    trend = _signal_trend(weekly)
    gain_by_vol = _gain_by_volatility(matching, config)

    latest_ts = max((_parse(s["emitted_at"]) for s in matching if s.get("emitted_at")), default=None)
    cutoff_48h = (latest_ts - timedelta(hours=48)) if latest_ts else None
    recent = [
        s for s in matching
        if cutoff_48h and s.get("emitted_at") and _parse(s["emitted_at"]) >= cutoff_48h
    ]
    recent_gains = [s["outcome"]["gain_pct"] for s in recent]
    recent_avg_gain_pct = sum(recent_gains) / len(recent_gains) if recent_gains else 0.0

    with_24h = [s["outcome"] for s in matching if "gain_24h_pct" in s["outcome"]]
    avg_gain_24h = sum(o["gain_24h_pct"] for o in with_24h) / len(with_24h) if with_24h else 0.0
    max_gain_24h = max(o["max_gain_24h_pct"] for o in with_24h) if with_24h else 0.0

    # Evaluation span in days (from first to last evaluated signal)
    timestamps = [s["emitted_at"] for s in matching if s.get("emitted_at")]
    evaluation_days = 0
    if len(timestamps) >= 2:
        evaluation_days = (_parse(max(timestamps)) - _parse(min(timestamps))).days

    # Score: avg_gain_pct normalised to [0,1] where 0 = -10%, 0.5 = 0%, 1.0 = +10%
    score = round(max(0.0, min(1.0, (avg_gain_pct + 0.10) / 0.20)), 4)

    # Deprecation: time-aware
    # - candidate  : not enough signals yet
    # - short eval : deprecate only on severe loss (< rule_early_deprecation_gain)
    # - mature eval : deprecate on zero or below-zero avg gain
    avg_transaction_gain = transaction_gains.get(rule_id, 0.0)
    mature = evaluation_days >= config.rule_mature_days
    if signal_count < config.rule_min_signals:
        status = "candidate"
    elif mature and avg_transaction_gain <= config.rule_mature_deprecation_gain:
        status = "deprecate"
    elif not mature and avg_gain_pct < config.rule_early_deprecation_gain:
        status = "deprecate"
    else:
        status = "active"

    return RuleScore(
        rule_id=rule_id,
        description=description,
        signal_count=signal_count,
        evaluation_days=evaluation_days,
        avg_gain_pct=avg_gain_pct,
        recent_avg_gain_pct=recent_avg_gain_pct,
        avg_transaction_gain=avg_transaction_gain,
        positive_rate=positive_rate,
        avg_gain_24h=avg_gain_24h,
        max_gain_24h=max_gain_24h,
        min_gain_pct=min_gain_pct,
        p25_gain_pct=p25_gain_pct,
        p75_gain_pct=p75_gain_pct,
        longest_win_streak=longest_win_streak,
        longest_loss_streak=longest_loss_streak,
        weekly_signal_counts=weekly,
        signal_trend=trend,
        avg_gain_by_volatility=gain_by_vol,
        score=score,
        status=status,
    )
