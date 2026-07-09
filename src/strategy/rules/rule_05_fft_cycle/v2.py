"""Rule 05 — Enhanced FFT Cycle with Momentum Confirmation (v2).

This modification to rule_05_fft_cycle_v1 aims to improve signal quality and
positive rates by adding a momentum-based filtering mechanism. While the
original rule identifies cyclical troughs and peaks using FFT, this enhanced
version will only generate a BuySignal if the detected cycle trough is
confirmed by a short-to-medium term upward momentum, and a SellSignal if
the detected cycle peak is confirmed by a short-to-medium term downward
momentum. This filters out cycle signals that go against the prevailing trend,
reducing false positives and increasing the probability of profitable trades.
"""

from __future__ import annotations

import cmath
import math
import statistics
import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


# Minimum warm candles; fewer than half a 24-hour window is not enough for cycle detection
# This constant is effectively overridden by SMA_MEDIUM_PERIOD (20) for the combined rule.
MIN_CANDLES_FFT = 12

# Peak amplitude of the dominant cycle must be at least this fraction of price std
AMPLITUDE_THRESHOLD = 0.3

# cos(phase) must be below −TROUGH_THRESHOLD to be considered a trough
TROUGH_THRESHOLD = 0.7

# Momentum indicator periods (adapted from pseudocode due to 24-candle warm data limit).
# The pseudocode suggested 20-period and 50-period SMAs. Given 'warm' data is capped
# at 24 hourly candles, we use 10-period for short-term and 20-period for medium-term
# to ensure computability within available data.
SMA_SHORT_PERIOD = 10
SMA_MEDIUM_PERIOD = 20


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
        # Ensure enough warm candles for both FFT and momentum calculations.
        # The combined requirement is the maximum of MIN_CANDLES_FFT and SMA_MEDIUM_PERIOD.
        min_required_candles = max(MIN_CANDLES_FFT, SMA_MEDIUM_PERIOD)
        if len(pair_data.warm) < min_required_candles or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        n_prices = len(prices)

        # 1. Calculate detrended price data.
        detrended = _detrend(prices)
        n_detrended = len(detrended)

        # 2. Apply Discrete Fourier Transform (DFT) to identify dominant cycle and its phase.
        X = _dft(detrended)

        # Find dominant frequency k_star (excluding DC component k=0)
        # We only need to check up to n_detrended // 2 for real signals.
        if n_detrended // 2 < 1:  # Not enough data for meaningful frequency analysis
            continue
        
        k_star = max(range(1, n_detrended // 2 + 1), key=lambda k: abs(X[k]))
        amplitude = 2 * abs(X[k_star]) / n_detrended

        price_std = statistics.stdev(prices)
        if price_std == 0 or amplitude / price_std < AMPLITUDE_THRESHOLD:
            continue

        # Calculate phase of the dominant cycle at the *end* of the series (index n-1)
        phase = 2 * math.pi * k_star * (n_detrended - 1) / n_detrended + cmath.phase(X[k_star])
        cos_phase = math.cos(phase)
        
        # Get current timestamp and price from the hot data (most recent tick)
        ts = pair_data.hot[-1].polled_at
        current_price = pair_data.hot[-1].last_price

        # 3. Calculate short-term and medium-term momentum indicators (SMAs).
        # We've already ensured len(prices) >= SMA_MEDIUM_PERIOD.
        sma_short = np.mean(prices[-SMA_SHORT_PERIOD:])
        sma_medium = np.mean(prices[-SMA_MEDIUM_PERIOD:])

        # 4. & 5. Generate potential BuySignal and Filter it with momentum.
        if cos_phase < -TROUGH_THRESHOLD:  # FFT indicates a trough
            # Filter BuySignal: Only emit if current price is above short SMA AND short SMA is above medium SMA.
            if current_price > sma_short and sma_short > sma_medium:
                signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        
        # 6. & 7. Generate potential SellSignal and Filter it with momentum.
        elif cos_phase > TROUGH_THRESHOLD:  # FFT indicates a peak
            # Filter SellSignal: Only emit if current price is below short SMA AND short SMA is below medium SMA.
            if current_price < sma_short and sma_short < sma_medium:
                signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals