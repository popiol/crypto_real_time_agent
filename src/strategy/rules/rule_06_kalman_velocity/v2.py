from __future__ import annotations

import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


MIN_TICKS = 30
MIN_WARM_CANDLES = 5

# How many recent ticks to search for a prior negative velocity
CROSS_WINDOW = 10

# Kalman noise parameters (tuned for ~1-second tick intervals)
_Q_P = 1e-4   # process noise on price position
_Q_V = 1e-2   # process noise on velocity (allows velocity to shift)
_R = 1.0      # measurement noise variance (tick price noise)

# New parameter: configurable percentage offset for price thresholds
# 0.005 means 0.5% offset
PRICE_OFFSET_FRACTION = 0.005


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
    if not prices:
        return []

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


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < MIN_TICKS or len(pair_data.warm) < MIN_WARM_CANDLES:
            continue

        prices = [t.last_price for t in ticks]
        velocities = _run_kalman(prices)
        
        if not velocities: # Handle case where _run_kalman returns empty list (e.g., if prices was empty)
            continue

        current_velocity = velocities[-1]
        recent_prior = velocities[-CROSS_WINDOW:-1]
        if not recent_prior: # Not enough history to detect a cross
            continue

        warm_mean = statistics.mean(c.close for c in pair_data.warm)
        current_price = ticks[-1].last_price
        ts = ticks[-1].polled_at

        # Calculate relaxed price thresholds
        buy_threshold = warm_mean * (1 - PRICE_OFFSET_FRACTION)
        sell_threshold = warm_mean * (1 + PRICE_OFFSET_FRACTION)

        # Buy signal: velocity just crossed negative to positive AND price is below the relaxed buy threshold
        if current_velocity > 0 and min(recent_prior) < 0 and current_price < buy_threshold:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Sell signal: velocity just crossed positive to negative AND price is above the relaxed sell threshold
        elif current_velocity < 0 and max(recent_prior) > 0 and current_price > sell_threshold:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals