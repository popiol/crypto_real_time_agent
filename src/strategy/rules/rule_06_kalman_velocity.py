"""Rule 06 — Signal processing: Kalman filter velocity reversal.

Runs a constant-velocity Kalman filter over the hot-tier tick prices to
produce a noise-filtered velocity (rate of price change) estimate at each
tick. Emits a buy signal when:
  1. The filtered velocity has just crossed from negative to positive
     (momentum reversal), AND
  2. The current price is below the warm-tier mean close (price is depressed).

State vector: x = [price, velocity]
Observation:  z = last_price  (scalar)

Transition:   F = [[1, 1],    Observation: H = [[1, 0]]
                   [0, 1]]

Process noise Q = diag(Q_P, Q_V), measurement noise R (scalar).

All 2×2 matrix operations are inlined as scalars to avoid any external
dependency. The derivation of each update equation is in the module docstring
of _run_kalman().
"""

from __future__ import annotations

import statistics

from src.agent.models import BuySignal, PairData

RULE_ID = "kalman_velocity_reversal"

MIN_TICKS = 30
MIN_WARM_CANDLES = 5

# How many recent ticks to search for a prior negative velocity
CROSS_WINDOW = 10

# Kalman noise parameters (tuned for ~1-second tick intervals)
_Q_P = 1e-4   # process noise on price position
_Q_V = 1e-2   # process noise on velocity (allows velocity to shift)
_R = 1.0      # measurement noise variance (tick price noise)

MarketData = dict[str, PairData]


def _run_kalman(prices: list[float]) -> list[float]:
    """Constant-velocity Kalman filter over a scalar price series.

    Returns the filtered velocity estimate v̂ at each time step.

    Predict:
        p̂⁻  = p̂ + v̂
        v̂⁻  = v̂
        P⁻   = F P Fᵀ + Q   (inlined below for the 2×2 case)

    Update (H = [[1, 0]], so only the first state component is observed):
        S    = P⁻[0,0] + R
        K    = P⁻[:,0] / S     → k0 = P⁻[0,0]/S, k1 = P⁻[1,0]/S
        p̂    = p̂⁻ + k0 · (z − p̂⁻)
        v̂    = v̂⁻ + k1 · (z − p̂⁻)
        P    = (I − K H) P⁻   (inlined below)
    """
    p = prices[0]
    v = 0.0
    # Covariance matrix P (start with high uncertainty)
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

    velocities: list[float] = [v]

    for z in prices[1:]:
        # --- Predict ---
        p_pr = p + v
        # F P Fᵀ for F = [[1,1],[0,1]]:
        pp00 = p00 + p01 + p10 + p11 + _Q_P
        pp01 = p01 + p11
        pp10 = p10 + p11
        pp11 = p11 + _Q_V

        # --- Update ---
        S = pp00 + _R
        k0 = pp00 / S
        k1 = pp10 / S
        innov = z - p_pr

        p = p_pr + k0 * innov
        v = v + k1 * innov

        p00 = (1 - k0) * pp00
        p01 = (1 - k0) * pp01
        p10 = pp10 - k1 * pp00
        p11 = pp11 - k1 * pp01

        velocities.append(v)

    return velocities


def kalman_velocity_reversal(data: MarketData) -> list[BuySignal]:
    signals: list[BuySignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < MIN_TICKS or len(pair_data.warm) < MIN_WARM_CANDLES:
            continue

        velocities = _run_kalman([t.last_price for t in ticks])

        # Condition 1: current velocity is positive (momentum turned upward)
        if velocities[-1] <= 0:
            continue

        # Condition 1 (cont.): velocity was negative within the recent window
        recent_prior = velocities[-CROSS_WINDOW:-1]
        if not recent_prior or min(recent_prior) >= 0:
            continue

        # Condition 2: current price is below warm-tier mean (price is depressed)
        warm_mean = statistics.mean(c.close for c in pair_data.warm)
        current_price = ticks[-1].last_price
        if current_price >= warm_mean:
            continue

        signals.append(
            BuySignal(
                pair=pair,
                rule_id=RULE_ID,
                timestamp=ticks[-1].polled_at,
                price=current_price,
            )
        )

    return signals
