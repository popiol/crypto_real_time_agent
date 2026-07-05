"""Rule 09 — Markov chain with trend confirmation (v2).

Extends v1 by adding a simple trend filter: the Markov-predicted direction
must agree with the short-term price trend (last 6 candles vs prior 6).
This reduces false signals in choppy, mean-reverting markets.
"""

from __future__ import annotations

import statistics

from src.agent.models import BuySignal, PairData, SellSignal

RULE_ID = "rule_09_markov_chain_v2"

N_STATES = 3
SIGNAL_THRESHOLD = 0.55
MIN_WARM_CANDLES = 14
TREND_WINDOW = 6

MarketData = dict[str, PairData]


def _discretize(closes: list[float]) -> list[int]:
    mean = statistics.mean(closes)
    if mean == 0:
        return [N_STATES // 2] * len(closes)
    deviations = [(c - mean) / mean for c in closes]
    lo, hi = min(deviations), max(deviations)
    span = hi - lo
    if span == 0:
        return [N_STATES // 2] * len(closes)
    states = []
    for d in deviations:
        s = int((d - lo) / span * N_STATES)
        states.append(min(s, N_STATES - 1))
    return states


def _transition_matrix(states: list[int]) -> list[list[float]]:
    counts = [[0] * N_STATES for _ in range(N_STATES)]
    for a, b in zip(states, states[1:]):
        counts[a][b] += 1
    T: list[list[float]] = []
    for row in counts:
        total = sum(row)
        T.append([c / total for c in row] if total > 0 else [0.0] * N_STATES)
    return T


def _trend_up(closes: list[float]) -> bool:
    mid = len(closes) // 2
    return statistics.mean(closes[mid:]) > statistics.mean(closes[:mid])


def enhance_markov_trend_confirmation(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        if len(pair_data.warm) < MIN_WARM_CANDLES or not pair_data.hot:
            continue

        closes = [c.close for c in pair_data.warm]
        states = _discretize(closes)
        T = _transition_matrix(states)

        current_state = states[-1]
        ts = pair_data.hot[-1].polled_at
        price = pair_data.hot[-1].last_price

        recent = closes[-TREND_WINDOW:] if len(closes) >= TREND_WINDOW else closes
        trend_bullish = _trend_up(recent)

        if current_state < N_STATES - 1:
            p_up = sum(T[current_state][j] for j in range(current_state + 1, N_STATES))
            if p_up > SIGNAL_THRESHOLD and trend_bullish:
                signals.append(BuySignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price))

        if current_state > 0:
            p_down = sum(T[current_state][j] for j in range(current_state))
            if p_down > SIGNAL_THRESHOLD and not trend_bullish:
                signals.append(SellSignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price))

    return signals
