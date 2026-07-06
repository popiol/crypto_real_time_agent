"""Rule 05 — Signal processing: FFT dominant cycle peak/trough detection.

Detrends the warm-tier hourly close prices, applies a DFT to identify the
dominant periodic cycle, and checks the current phase of that cycle.

    φ = 2π · k* · (N-1) / N + angle(X[k*])

Buy signal:  cos(φ) < −TROUGH_THRESHOLD  → cycle at trough, price likely to rise.
Sell signal: cos(φ) >  TROUGH_THRESHOLD  → cycle at peak, price likely to fall.

The dominant cycle must carry meaningful amplitude relative to price volatility
to filter out noise-driven detections.
"""

from __future__ import annotations

import cmath
import math
import statistics

from src.agent.models import BuySignal, PairData, SellSignal

RULE_ID = "rule_05_fft_cycle_v1"

# Minimum warm candles; fewer than half a 24-hour window is not enough for cycle detection
MIN_CANDLES = 12

# Peak amplitude of the dominant cycle must be at least this fraction of price std
AMPLITUDE_THRESHOLD = 0.3

# cos(phase) must be below −TROUGH_THRESHOLD to be considered a trough
TROUGH_THRESHOLD = 0.7

MarketData = dict[str, PairData]


def _detrend(series: list[float]) -> list[float]:
    """Remove the least-squares linear trend so the DFT reflects cycles, not drift."""
    n = len(series)
    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(series)
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, series))
    den = sum((xi - x_mean) ** 2 for xi in xs)
    slope = num / den if den else 0.0
    intercept = y_mean - slope * x_mean
    return [yi - (slope * xi + intercept) for xi, yi in zip(xs, series)]


def _dft(series: list[float]) -> list[complex]:
    """Naive O(n²) DFT — correct and fast enough for the ≤24-point warm-tier window."""
    n = len(series)
    return [
        sum(series[t] * cmath.exp(-2j * math.pi * k * t / n) for t in range(n))
        for k in range(n)
    ]


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        if len(pair_data.warm) < MIN_CANDLES or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        detrended = _detrend(prices)
        n = len(detrended)

        X = _dft(detrended)

        k_star = max(range(1, n // 2 + 1), key=lambda k: abs(X[k]))
        amplitude = 2 * abs(X[k_star]) / n

        price_std = statistics.stdev(prices)
        if price_std == 0 or amplitude / price_std < AMPLITUDE_THRESHOLD:
            continue

        phase = 2 * math.pi * k_star * (n - 1) / n + cmath.phase(X[k_star])
        cos_phase = math.cos(phase)
        ts = pair_data.hot[-1].polled_at
        price = pair_data.hot[-1].last_price

        if cos_phase < -TROUGH_THRESHOLD:
            signals.append(BuySignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price))
        elif cos_phase > TROUGH_THRESHOLD:
            signals.append(SellSignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price))

    return signals
