"""Rule b4c16abf-143d-4f82-8c1a-a3bb40f92812 — Enhanced FFT Cycle with VWAP Confirmation Filter.

This rule refines 'rule_05_fft_cycle_v1' by integrating a Volume-Weighted Average Price
(VWAP) confirmation. It will only emit a Buy signal if the FFT cycle indicates a trough
AND the current price is significantly below the recent VWAP, suggesting undervaluation
with strong buying interest. Conversely, a Sell signal will only be emitted if the FFT
cycle indicates a peak AND the current price is significantly above the recent VWAP,
suggesting overvaluation with strong selling interest. This aims to filter out
low-conviction cycle signals and improve profitability per trade.
"""

from __future__ import annotations

import cmath
import math
import statistics

from src.agent.models import BuySignal, HistoricalCandle, MarketData, SellSignal

# --- FFT Cycle Detection Constants (from original rule_05_fft_cycle_v1) ---
# Minimum warm candles; fewer than half a 24-hour window is not enough for cycle detection
MIN_CANDLES = 12

# Peak amplitude of the dominant cycle must be at least this fraction of price std
AMPLITUDE_THRESHOLD = 0.3

# cos(phase) must be below −TROUGH_THRESHOLD to be considered a trough
TROUGH_THRESHOLD = 0.7

# --- VWAP Confirmation Filter Constants ---
# Number of recent warm candles to use for VWAP calculation.
# For hourly candles, 12 means a 12-hour VWAP.
VWAP_LOOKBACK_PERIOD = 12

# Percentage deviation from VWAP required for confirmation.
# E.g., 0.005 means 0.5% below VWAP for buy, 0.5% above for sell.
VWAP_DEVIATION_THRESHOLD = 0.005


# --- FFT Helper Functions (from original rule_05_fft_cycle_v1) ---
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


# --- VWAP Helper Function ---
def _calculate_vwap(candles: list[HistoricalCandle], lookback_period: int) -> float | None:
    """
    Calculates the Volume-Weighted Average Price (VWAP) for a given list of candles.

    Args:
        candles: A list of HistoricalCandle objects.
        lookback_period: The number of most recent candles to consider for VWAP calculation.

    Returns:
        The calculated VWAP as a float, or None if VWAP cannot be calculated
        (e.g., insufficient data or zero total volume).
    """
    if not candles:
        return None

    # Ensure we don't try to look back further than available data
    start_index = max(0, len(candles) - lookback_period)
    relevant_candles = candles[start_index:]

    if not relevant_candles:
        return None

    total_pv = 0.0  # Sum of (price * volume)
    total_volume = 0.0  # Sum of volume

    for candle in relevant_candles:
        total_pv += candle.close * candle.volume
        total_volume += candle.volume

    if total_volume == 0:
        return None  # Cannot calculate VWAP if there's no trading volume

    return total_pv / total_volume


# --- Main Signal Function ---
def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on an enhanced FFT cycle detection with VWAP confirmation.

    A Buy signal is emitted if the FFT cycle indicates a trough AND the current price
    is significantly below the recent VWAP.
    A Sell signal is emitted if the FFT cycle indicates a peak AND the current price
    is significantly above the recent VWAP.

    Args:
        data: MarketData object containing warm (historical) and hot (real-time) data.

    Returns:
        A list of BuySignal or SellSignal objects.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # 1. Initial data availability check
        if len(pair_data.warm) < MIN_CANDLES or not pair_data.hot:
            continue

        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # 2. Calculate current VWAP
        vwap = _calculate_vwap(pair_data.warm, VWAP_LOOKBACK_PERIOD)
        if vwap is None:
            # If VWAP cannot be calculated (e.g., no volume), skip this pair
            continue

        # 3. Obtain FFT cycle signal from rule_05_fft_cycle_v1 logic
        prices_for_fft = [c.close for c in pair_data.warm]
        detrended = _detrend(prices_for_fft)
        n = len(detrended)

        X = _dft(detrended)

        # Find the dominant cycle (excluding DC component k=0)
        # Use n // 2 + 1 to include the Nyquist frequency if n is even
        k_star = max(range(1, n // 2 + 1), key=lambda k: abs(X[k]))
        amplitude = 2 * abs(X[k_star]) / n

        # Filter out noise-driven detections: dominant cycle must have meaningful amplitude
        price_std = statistics.stdev(prices_for_fft)
        if price_std == 0 or amplitude / price_std < AMPLITUDE_THRESHOLD:
            continue  # FFT signal not strong enough relative to price volatility

        # Calculate the phase of the dominant cycle at the current moment (n-1)
        # 2π · k* · (N-1) / N + angle(X[k*])
        phase = 2 * math.pi * k_star * (n - 1) / n + cmath.phase(X[k_star])
        cos_phase = math.cos(phase)

        # Determine potential FFT signal (peak or trough)
        fft_indicates_trough = cos_phase < -TROUGH_THRESHOLD
        fft_indicates_peak = cos_phase > TROUGH_THRESHOLD

        # 4. Apply VWAP filter to the FFT signal
        if fft_indicates_trough:
            # FFT indicates a trough, confirm with price significantly below VWAP
            if current_price < (vwap * (1 - VWAP_DEVIATION_THRESHOLD)):
                signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif fft_indicates_peak:
            # FFT indicates a peak, confirm with price significantly above VWAP
            if current_price > (vwap * (1 + VWAP_DEVIATION_THRESHOLD)):
                signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals