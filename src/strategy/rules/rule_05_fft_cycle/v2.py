from __future__ import annotations

import cmath
import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# --- Rule Parameters ---
FFT_WINDOW_N = 24  # Number of hourly candles for FFT analysis (e.g., 24 hours)
SMA_LONG_WINDOW_M = 72  # Number of hourly candles for long-term SMA (e.g., 72 hours = 3 days)
AMPLITUDE_THRESHOLD = 0.3  # Min dominant cycle amplitude relative to price std
FFT_THRESHOLD = 0.7  # cos(phase) must be below -FFT_THRESHOLD for trough, above FFT_THRESHOLD for peak
TREND_SLOPE_THRESHOLD = 0.001  # Max absolute percentage change in SMA for 'flat' trend (e.g., 0.1%)

# Minimum warm candles required for analysis
MIN_CANDLES_REQUIRED = max(FFT_WINDOW_N, SMA_LONG_WINDOW_M)


# --- Helper Functions ---

def _sma(series: list[float], window: int) -> list[float]:
    """Calculates a trailing Simple Moving Average (SMA)."""
    if len(series) < window:
        return []
    sma_values = []
    for i in range(len(series) - window + 1):
        sma_values.append(statistics.mean(series[i : i + window]))
    return sma_values

def _detrend_mean(series: list[float]) -> list[float]:
    """Removes the mean from a series, as suggested by `prices - SMA(prices, N)`
    when `N` is the length of `prices` for FFT analysis, effectively removing the DC component."""
    if not series:
        return []
    mean_val = statistics.mean(series)
    return [x - mean_val for x in series]

def _dft(series: list[float]) -> list[complex]:
    """Naive O(n²) DFT — correct and fast enough for typical warm-tier windows."""
    n = len(series)
    if n == 0:
        return []
    return [
        sum(series[t] * cmath.exp(-2j * math.pi * k * t / n) for t in range(n))
        for k in range(n)
    ]

def _reconstruct_dominant_cycle(amplitude: float, k_star: int, n_points: int, fft_coeff_phase: float) -> list[float]:
    """
    Reconstructs the dominant cycle as a sine wave.
    `fft_coeff_phase` is the phase of the DFT coefficient X[k_star], representing phase at t=0.
    """
    reconstructed = []
    for t in range(n_points):
        # A_k * cos(2 * pi * k * t / N + phi_k)
        reconstructed.append(amplitude * math.cos(2 * math.pi * k_star * t / n_points + fft_coeff_phase))
    return reconstructed


# --- Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm data and hot data for current price/timestamp
        if len(pair_data.warm) < MIN_CANDLES_REQUIRED or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        current_ts = pair_data.hot[-1].polled_at
        current_price = pair_data.hot[-1].last_price

        # 1. Prepare data for FFT analysis
        # Use the last N candles for FFT
        fft_analysis_prices = prices[-FFT_WINDOW_N:]
        if len(fft_analysis_prices) < FFT_WINDOW_N:
            continue # Not enough data even after slicing

        detrended_fft_prices = _detrend_mean(fft_analysis_prices)
        n = len(detrended_fft_prices) # Should be FFT_WINDOW_N

        # 2. Apply FFT to detrended prices
        X = _dft(detrended_fft_prices)
        if not X:
            continue

        # 3. Extract dominant cycle
        # Find the dominant frequency (excluding DC component k=0)
        # Using range(1, n // 2 + 1) to consider positive frequencies.
        # Max over abs(X[k]) gives dominant component.
        try:
            k_star = max(range(1, n // 2 + 1), key=lambda k: abs(X[k]))
        except ValueError: # range might be empty if n <= 1
            continue

        amplitude = 2 * abs(X[k_star]) / n
        fft_coeff_phase = cmath.phase(X[k_star]) # Phase of the dominant DFT coefficient

        # Filter out noise-driven detections: dominant cycle amplitude must be significant
        # Calculate standard deviation over the FFT analysis window prices
        price_std = statistics.stdev(fft_analysis_prices) if len(fft_analysis_prices) > 1 else 0.0
        if price_std == 0 or amplitude / price_std < AMPLITUDE_THRESHOLD:
            continue

        # 4. Reconstruct dominant cycle
        reconstructed_cycle = _reconstruct_dominant_cycle(amplitude, k_star, n, fft_coeff_phase)
        if not reconstructed_cycle:
            continue

        # 5. Get current cycle position (normalized)
        cycle_max_abs = max(abs(val) for val in reconstructed_cycle)
        current_cycle_position = reconstructed_cycle[-1] / cycle_max_abs if cycle_max_abs != 0 else 0.0

        # 6. Calculate long-term trend using SMA
        long_term_sma_values = _sma(prices, SMA_LONG_WINDOW_M)
        if len(long_term_sma_values) < 2:
            continue # Not enough SMA values to calculate slope

        # Calculate trend slope as percentage change of the SMA
        prev_sma = long_term_sma_values[-2]
        current_sma = long_term_sma_values[-1]

        # Handle cases where prev_sma might be zero (unlikely for prices, but good practice)
        if prev_sma == 0:
            trend_slope = 0.0
        else:
            trend_slope = (current_sma - prev_sma) / prev_sma

        # 7. Generate signal with trend confirmation
        if current_cycle_position <= -FFT_THRESHOLD:  # Cycle indicates a trough (potential Buy)
            if trend_slope >= -TREND_SLOPE_THRESHOLD:  # Trend is flat or upward (or very slightly downward within threshold)
                signals.append(BuySignal(pair=pair, timestamp=current_ts, price=current_price))
        elif current_cycle_position >= FFT_THRESHOLD:  # Cycle indicates a peak (potential Sell)
            if trend_slope <= TREND_SLOPE_THRESHOLD:  # Trend is flat or downward (or very slightly upward within threshold)
                signals.append(SellSignal(pair=pair, timestamp=current_ts, price=current_price))

    return signals