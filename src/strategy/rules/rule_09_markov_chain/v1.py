"""Rule 09 — Markov chain: price-level transition probability.

Discretises the warm-tier hourly close prices into N_STATES levels relative
to the series mean, builds an empirical transition probability matrix from
the sequential state pairs.

Buy signal:  P(move to higher state) > SIGNAL_THRESHOLD.
Sell signal: P(move to lower state)  > SIGNAL_THRESHOLD.

N_STATES is kept small (3) because the warm tier provides at most 24 data
points; more states produce a matrix that is too sparse to be reliable.

States are assigned by equal-width bins over the range of percentage
deviations from the mean:
    state 0 = below-average price range
    state 1 = near-average
    state 2 = above-average
"""

from __future__ import annotations

import statistics

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


N_STATES = 3
SIGNAL_THRESHOLD = 0.6      # P(move to higher state) must exceed this
MIN_WARM_CANDLES = 12       # need enough transitions to estimate probabilities



def _discretize(closes: list[float]) -> list[int]:
    """Assign each close price to a state in [0, N_STATES-1].

    Bins are equal-width over the range of percentage deviations from the mean.
    """
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
    """Build a row-normalised transition probability matrix from state sequence."""
    counts = [[0] * N_STATES for _ in range(N_STATES)]
    for a, b in zip(states, states[1:]):
        counts[a][b] += 1

    T: list[list[float]] = []
    for row in counts:
        total = sum(row)
        T.append([c / total for c in row] if total > 0 else [0.0] * N_STATES)
    return T


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
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

        if current_state < N_STATES - 1:
            p_up = sum(T[current_state][j] for j in range(current_state + 1, N_STATES))
            if p_up > SIGNAL_THRESHOLD:
                signals.append(BuySignal(pair=pair, timestamp=ts, price=price))

        if current_state > 0:
            p_down = sum(T[current_state][j] for j in range(current_state))
            if p_down > SIGNAL_THRESHOLD:
                signals.append(SellSignal(pair=pair, timestamp=ts, price=price))

    return signals
