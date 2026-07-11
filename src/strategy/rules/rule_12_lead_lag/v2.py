"""Rule 12 v2 — Lead-lag: momentum leaders signal laggards.

Algorithm
---------
1. Find all assets whose last-hour price change is above LEADER_CHANGE_MIN,
   sorted by that change descending (strongest movers first).
2. For each such leader X, scan every other asset Y and compute the
   Pearson correlation:

       corr(X_closes[-24:-1], Y_closes[-23:])

   This compares X's recent history (excluding the last candle) against
   Y's most recent window, measuring how closely Y is tracking X with a
   one-candle lag.

3. Take the Y with the highest such correlation. If correlation > CORR_MIN
   and Y's last-hour change is below LAGGARD_CHANGE_MAX → BuySignal for Y.
   Stop after the first buy signal (at most one per cycle).

4. Independently, emit a SellSignal for every asset whose last-hour change
   is above LAGGARD_SELL_MIN.
"""

from __future__ import annotations

import math

import numpy as np

from src.agent.models import BuySignal, MarketData, SellSignal

LEADER_CHANGE_MIN: float = 0.01  # X must have risen ≥1 % in the last hour
CORR_MIN: float = 0.5  # minimum correlation to treat Y as a follower
LAGGARD_CHANGE_MAX: float = 0.001  # Y has not yet moved — buy threshold
LAGGARD_SELL_MIN: float = 0.005  # independent sell threshold for any asset
MIN_WARM_CANDLES: int = 24  # need full 24-candle window for both assets


# ── Helpers ────────────────────────────────────────────────────────────────────


def _hour_change(closes: list[float]) -> float:
    """Percentage change of the last warm candle relative to the one before it."""
    if len(closes) < 2 or closes[-2] == 0:
        return 0.0
    return (closes[-1] - closes[-2]) / closes[-2]


def _correlation(x: list[float], y: list[float]) -> float:
    """corr(X[-24:-1], Y[-23:]) — both slices have length 23."""
    xa = np.array(x[-24:-1], dtype=np.float64)  # 23 elements
    ya = np.array(y[-23:], dtype=np.float64)  # 23 elements
    if len(xa) < 2 or len(xa) != len(ya):
        return 0.0
    c = float(np.corrcoef(xa, ya)[0, 1])
    return c if math.isfinite(c) else 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────


def _best_follower(
    leader: str, closes: dict[str, list[float]]
) -> tuple[str, float] | None:
    """Return (best_follower_pair, correlation) or None if none qualifies."""
    best_pair: str | None = None
    best_corr: float = CORR_MIN

    for candidate, candidate_closes in closes.items():
        if candidate == leader:
            continue
        corr = _correlation(closes[leader], candidate_closes)
        if corr > best_corr:
            best_corr = corr
            best_pair = candidate

    return (best_pair, best_corr) if best_pair is not None else None


# ── Signal generation ──────────────────────────────────────────────────────────


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    # Build per-asset close lists for assets that have enough warm candles
    closes: dict[str, list[float]] = {
        pair: [c.close for c in pd.warm]
        for pair, pd in data.items()
        if len(pd.warm) >= MIN_WARM_CANDLES
    }

    # Step 1 — leaders: last-hour change > LEADER_CHANGE_MIN, sorted descending
    leaders = sorted(
        [
            (pair, _hour_change(cl))
            for pair, cl in closes.items()
            if _hour_change(cl) > LEADER_CHANGE_MIN
        ],
        key=lambda t: t[1],
        reverse=True,
    )

    signals: list[BuySignal | SellSignal] = []

    # Step 4 — sell signals: independent of correlation, all qualifying assets
    for pair, cl in closes.items():
        if _hour_change(cl) > LAGGARD_SELL_MIN:
            pd = data[pair]
            if pd.hot:
                tick = pd.hot[-1]
                signals.append(
                    SellSignal(
                        pair=pair,
                        timestamp=tick.polled_at,
                        price=tick.last_price,
                        confidence=1.0,
                    )
                )

    # Steps 2 & 3 — for each leader find the most correlated laggard
    for leader, _leader_chg in leaders:
        result = _best_follower(leader, closes)
        if result is None:
            continue

        follower, corr = result
        if _hour_change(closes[follower]) >= LAGGARD_CHANGE_MAX:
            continue

        pd_follower = data[follower]
        if not pd_follower.hot:
            continue

        tick = pd_follower.hot[-1]
        signals.append(
            BuySignal(
                pair=follower,
                timestamp=tick.polled_at,
                price=tick.last_price,
                confidence=corr,
            )
        )
        break  # at most one buy signal per cycle

    return signals
