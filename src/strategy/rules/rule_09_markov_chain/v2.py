from __future__ import annotations

import statistics
from datetime import datetime

import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle


_RULE_ID = "965092b1-24c1-48de-ac1b-cfd8efd574e4"

# Markov Chain Parameters (from original rule_09)
_N_STATES = 3
_MIN_WARM_CANDLES_MC = 12  # Minimum for Markov Chain itself

# New Rule Parameters (from pseudocode)
_STATE_TRANSITION_THRESHOLD = 0.60
_ATR_PERIOD = 14
_ATR_MULTIPLIER = 1.0
_RSI_PERIOD = 14
_RSI_BUY_THRESHOLD = 50
_RSI_SELL_THRESHOLD = 50

# Calculate the overall minimum candles required for all indicators
# 1. Markov Chain needs _MIN_WARM_CANDLES_MC
# 2. RSI needs _RSI_PERIOD + 1 candles for the first RSI value to be calculated.
# 3. ATR needs sufficient candles to calculate the current ATR AND the average of _ATR_PERIOD * 2 previous ATRs.
#    - To get (_ATR_PERIOD * 2) previous ATR values + the current ATR value = (_ATR_PERIOD * 2 + 1) ATR values in total.
#    - Since `_calculate_atr` returns `len(closes) - period + 1` values (after initial smoothing period),
#      we need `len(closes) - _ATR_PERIOD + 1 >= (_ATR_PERIOD * 2 + 1)`.
#    - This simplifies to `len(closes) >= _ATR_PERIOD * 3`.
_MIN_CANDLES_REQUIRED = max(_MIN_WARM_CANDLES_MC, _RSI_PERIOD + 1, _ATR_PERIOD * 3)


def _discretize(closes: list[float]) -> list[int]:
    """Assign each close price to a state in [0, _N_STATES-1].

    Bins are equal-width over the range of percentage deviations from the mean.
    """
    mean = statistics.mean(closes)
    if mean == 0:
        return [_N_STATES // 2] * len(closes)

    deviations = [(c - mean) / mean for c in closes]
    lo, hi = min(deviations), max(deviations)
    span = hi - lo
    if span == 0:
        return [_N_STATES // 2] * len(closes)

    states = []
    for d in deviations:
        s = int((d - lo) / span * _N_STATES)
        states.append(min(s, _N_STATES - 1))
    return states


def _transition_matrix(states: list[int]) -> list[list[float]]:
    """Build a row-normalised transition probability matrix from state sequence."""
    counts = [[0] * _N_STATES for _ in range(_N_STATES)]
    for a, b in zip(states, states[1:]):
        counts[a][b] += 1

    T: list[list[float]] = []
    for row in counts:
        total = sum(row)
        T.append([c / total for c in row] if total > 0 else [0.0] * _N_STATES)
    return T


def _calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Calculates Average True Range (ATR) using Wilder's smoothing."""
    if len(closes) < period:
        return np.array([])

    tr_values = []
    for i in range(len(closes)):
        if i == 0:
            # For the very first candle, TR is simply high - low
            tr = highs[i] - lows[i]
        else:
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_values.append(tr)
    tr_values = np.array(tr_values)

    if len(tr_values) < period:
        return np.array([])

    atrs = np.zeros_like(tr_values)
    # Initial ATR is SMA of the first 'period' TR values
    atrs[period - 1] = np.mean(tr_values[:period])

    # Calculate subsequent ATR values using Wilder's smoothing
    for i in range(period, len(tr_values)):
        atrs[i] = (atrs[i-1] * (period - 1) + tr_values[i]) / period

    return atrs[period-1:] # Return ATR values starting from the first full period


def _calculate_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Calculates Relative Strength Index (RSI) using Wilder's smoothing."""
    if len(closes) < period + 1:
        return np.array([])

    diff = np.diff(closes)
    gains = np.where(diff > 0, diff, 0)
    losses = np.where(diff < 0, abs(diff), 0)

    avg_gains = np.zeros_like(gains)
    avg_losses = np.zeros_like(losses)

    # Initial average gain/loss (SMA of first 'period' values)
    avg_gains[period-1] = np.mean(gains[:period])
    avg_losses[period-1] = np.mean(losses[:period])

    # Subsequent averages using Wilder's smoothing
    for i in range(period, len(gains)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i]) / period

    # Calculate Relative Strength (RS)
    # Handle division by zero for avg_losses
    rs = np.divide(avg_gains[period-1:], avg_losses[period-1:],
                   out=np.full_like(avg_gains[period-1:], np.inf), # np.inf if avg_losses is 0
                   where=avg_losses[period-1:] != 0)

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm candle data for all calculations
        if len(pair_data.warm) < _MIN_CANDLES_REQUIRED or not pair_data.hot:
            continue

        # Extract data for calculations
        closes = [c.close for c in pair_data.warm]
        highs = [c.high for c in pair_data.warm]
        lows = [c.low for c in pair_data.warm]

        # Convert to numpy arrays for efficient indicator calculations
        closes_np = np.array(closes)
        highs_np = np.array(highs)
        lows_np = np.array(lows)

        # 1. Calculate Markov Chain State Transition Probabilities
        states = _discretize(closes)
        T = _transition_matrix(states)

        current_state = states[-1]
        p_up = 0.0
        if current_state < _N_STATES - 1:
            p_up = sum(T[current_state][j] for j in range(current_state + 1, _N_STATES))

        p_down = 0.0
        if current_state > 0:
            p_down = sum(T[current_state][j] for j in range(current_state))

        # 2. Calculate Average True Range (ATR)
        atrs = _calculate_atr(highs_np, lows_np, closes_np, _ATR_PERIOD)
        # We need at least (_ATR_PERIOD * 2 + 1) ATR values to calculate current and average ATR
        if len(atrs) < (_ATR_PERIOD * 2 + 1):
            continue

        current_atr = atrs[-1]
        # AVERAGE_ATR is the SMA of the previous _ATR_PERIOD * 2 ATR values, excluding the current one
        average_atr = np.mean(atrs[-(_ATR_PERIOD * 2 + 1):-1])

        # 3. Calculate Relative Strength Index (RSI)
        rsi_values = _calculate_rsi(closes_np, _RSI_PERIOD)
        # Ensure RSI values were successfully calculated
        if len(rsi_values) == 0:
            continue
        current_rsi = rsi_values[-1]

        # Get latest timestamp and price for signal
        ts = pair_data.hot[-1].polled_at
        price = pair_data.hot[-1].last_price

        # 4. Generate Signal based on combined conditions
        # BUY condition
        if (p_up > _STATE_TRANSITION_THRESHOLD and
            current_atr > average_atr * _ATR_MULTIPLIER and
            current_rsi > _RSI_BUY_THRESHOLD):
            signals.append(BuySignal(pair=pair, timestamp=ts, price=price, rule_id=_RULE_ID))

        # SELL condition
        elif (p_down > _STATE_TRANSITION_THRESHOLD and
              current_atr > average_atr * _ATR_MULTIPLIER and
              current_rsi < _RSI_SELL_THRESHOLD):
            signals.append(SellSignal(pair=pair, timestamp=ts, price=price, rule_id=_RULE_ID))

    return signals