from __future__ import annotations

import cmath
import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# Minimum warm candles; fewer than half a 24-hour window is not enough for cycle detection
MIN_CANDLES = 12

# Peak amplitude of the dominant cycle must be at least this fraction of price std
AMPLITUDE_THRESHOLD = 0.3

# cos(phase) must be below −TROUGH_THRESHOLD to be considered a trough
TROUGH_THRESHOLD = 0.7

# Period for the long-term Exponential Moving Average (EMA)
EMA_PERIOD = 12 # 12 hourly candles for a half-day EMA

# Rule identifier
RULE_ID = "rule_fft_ema_v1"


def _detrend(series: list[float]) -> list[float]:
    """Remove the least-squares linear trend so the DFT reflects cycles, not drift."""
    n = len(series)
    if n < 2:
        return [0.0] * n # Cannot detrend with less than 2 points

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
    if n == 0:
        return []
    return [
        sum(series[t] * cmath.exp(-2j * math.pi * k * t / n) for t in range(n))
        for k in range(n)
    ]


def _ema_last_value(prices: list[float], period: int) -> float | None:
    """
    Calculates the last value of the Exponential Moving Average (EMA) for a given series of prices.
    Returns None if there is insufficient data.
    """
    if len(prices) < period:
        return None

    k = 2 / (period + 1)

    # Calculate initial SMA for the first 'period' values
    ema = sum(prices[:period]) / period

    # Calculate EMA for the rest of the prices
    for i in range(period, len(prices)):
        ema = (prices[i] * k) + (ema * (1 - k))
    
    return ema


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm candles for both FFT and EMA, and at least one hot tick
        required_candles = max(MIN_CANDLES, EMA_PERIOD)
        if len(pair_data.warm) < required_candles or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        n = len(prices)

        # 1. Detrend price data
        detrended = _detrend(prices)

        # 2. Apply FFT to detrended prices
        # 3. Identify dominant cycle and its phase
        X = _dft(detrended)
        if not X: # Handle empty DFT result if input was empty
            continue

        # Find the dominant frequency (excluding DC component k=0)
        # We only care about k from 1 to n//2 for real-valued signals
        k_star = max(range(1, n // 2 + 1), key=lambda k: abs(X[k]))

        # Calculate amplitude and filter out noise-driven detections
        amplitude = 2 * abs(X[k_star]) / n
        price_std = statistics.stdev(prices) if len(prices) > 1 else 0.0

        if price_std == 0 or amplitude / price_std < AMPLITUDE_THRESHOLD:
            continue

        # Calculate the phase of the dominant cycle
        # φ = 2π · k* · (N-1) / N + angle(X[k*])
        phase = 2 * math.pi * k_star * (n - 1) / n + cmath.phase(X[k_star])
        cos_phase = math.cos(phase)

        # 4. Calculate long-term EMA
        long_term_ema = _ema_last_value(prices, EMA_PERIOD)
        if long_term_ema is None: # Should be caught by initial required_candles check, but good to be safe
            continue

        # Get current price and timestamp from the latest hot tick
        ts = pair_data.hot[-1].polled_at
        current_price = pair_data.hot[-1].last_price

        # 5. Generate signals with EMA trend confirmation
        # A trough is when cos(phase) is near -1 (e.g., < -TROUGH_THRESHOLD)
        # A peak is when cos(phase) is near 1 (e.g., > TROUGH_THRESHOLD)
        if cos_phase < -TROUGH_THRESHOLD and current_price > long_term_ema:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price, rule_id=RULE_ID))
        elif cos_phase > TROUGH_THRESHOLD and current_price < long_term_ema:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price, rule_id=RULE_ID))

    return signals