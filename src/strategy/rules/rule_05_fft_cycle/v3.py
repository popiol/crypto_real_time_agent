from __future__ import annotations

import cmath
import math
import statistics
from collections import deque

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle


# --- Constants for the Enhanced FFT Cycle Rule ---
# Minimum warm candles required for all calculations (FFT, EMA, ATR).
# This value is adjusted from the pseudocode's implied window sizes (256, 200)
# to be compatible with the `pair_data.warm` list, which has "At most 24 entries".
# A value of 20 allows for an ATR(14) calculation and an EMA(15) calculation
# to yield at least a few data points for their respective series.
MIN_CANDLES = 20

# FFT Cycle Detection Parameters
# The FFT_WINDOW will be dynamic, using the available length of prices (up to 24)
# The DETREND_WINDOW logic is handled by the _detrend function over the full series.
AMPLITUDE_THRESHOLD = 0.3  # Peak amplitude of dominant cycle must be at least this fraction of price std
TROUGH_THRESHOLD = 0.7     # cos(phase) must be below −TROUGH_THRESHOLD for trough (BUY), above for peak (SELL)

# Trend Filter Parameters (adjusted for max 24 warm candles)
LONG_TERM_EMA_PERIOD = 15  # Period for long-term EMA trend filter

# Volatility Filter Parameters
ATR_PERIOD = 14            # Period for Average True Range
VOLATILITY_BAND_MULTIPLIER = 1.5 # Multiplier for historical ATR range for stability check
# The HISTORICAL_ATR_PERIOD for mean/std dev will be dynamic, using all available ATR values.


# --- Helper Functions ---

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
    """Naive O(n²) DFT — correct and fast enough for the <=24-point warm-tier window."""
    n = len(series)
    if n == 0:
        return []
    
    return [
        sum(series[t] * cmath.exp(-2j * math.pi * k * t / n) for t in range(n))
        for k in range(n)
    ]


def _calculate_ema(prices: list[float], period: int) -> list[float]:
    """Calculates Exponential Moving Average."""
    if len(prices) < period:
        return []
    
    ema_values = []
    smoothing_factor = 2 / (period + 1)

    # Initialize with SMA for the first 'period' values
    current_ema = statistics.mean(prices[:period])
    ema_values.append(current_ema)

    for i in range(period, len(prices)):
        current_ema = (prices[i] - current_ema) * smoothing_factor + current_ema
        ema_values.append(current_ema)
        
    return ema_values


def _calculate_atr(warm_candles: list[WarmCandle], period: int) -> list[float]:
    """Calculates Average True Range."""
    # Need at least period + 1 candles to get the first True Range and then ATR
    if len(warm_candles) < period + 1:
        return []

    true_ranges = []
    for i in range(1, len(warm_candles)):
        current_high = warm_candles[i].high
        current_low = warm_candles[i].low
        previous_close = warm_candles[i-1].close

        tr1 = current_high - current_low
        tr2 = abs(current_high - previous_close)
        tr3 = abs(current_low - previous_close)
        true_ranges.append(max(tr1, tr2, tr3))

    if len(true_ranges) < period:
        return []

    atr_values = []
    # Initial ATR is SMA of the first 'period' True Ranges
    current_atr = statistics.mean(true_ranges[:period])
    atr_values.append(current_atr)

    # Subsequent ATR values use EMA-like calculation (Wilder's smoothing)
    for i in range(period, len(true_ranges)):
        current_atr = (current_atr * (period - 1) + true_ranges[i]) / period
        atr_values.append(current_atr)

    return atr_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm candles for calculations and at least one hot tick for current price/timestamp.
        # MIN_CANDLES is set to 20 to allow for meaningful ATR(14) and EMA(15) calculations within the 24-candle limit.
        if len(pair_data.warm) < MIN_CANDLES or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        n = len(prices) # Actual window size for FFT and other calculations

        # --- 1. FFT Cycle Detection ---
        detrended = _detrend(prices)
        if not detrended:
             continue

        X = _dft(detrended)
        if not X:
            continue

        # Find dominant cycle (excluding DC component k=0 and Nyquist frequency k=n/2 if n is even).
        # Ensures there are enough frequency components to choose a dominant one.
        if (n // 2 + 1) <= 1:
            continue
        
        # Filter k_star to ensure it's within valid range for calculation, avoiding potential IndexError.
        valid_k_range = [k for k in range(1, n // 2 + 1) if k < len(X)]
        if not valid_k_range:
            continue
            
        k_star = max(valid_k_range, key=lambda k: abs(X[k]))
        
        amplitude = 2 * abs(X[k_star]) / n

        # Check amplitude threshold to filter out noise-driven detections.
        price_std = statistics.stdev(prices) if len(prices) > 1 else 0.0
        if price_std == 0 or amplitude / price_std < AMPLITUDE_THRESHOLD:
            continue

        # Calculate phase for the last point in the series (t = n-1).
        # The phase is normalized to be within [0, 2*pi).
        phase = (2 * math.pi * k_star * (n - 1) / n + cmath.phase(X[k_star])) % (2 * math.pi)
        cos_phase = math.cos(phase)

        # --- 2. Trend Filter (Long-term EMA) ---
        long_term_ema_series = _calculate_ema(prices, LONG_TERM_EMA_PERIOD)
        if not long_term_ema_series:
            continue
        
        current_ema = long_term_ema_series[-1]
        current_price = prices[-1]

        is_uptrend = current_price > current_ema
        is_downtrend = current_price < current_ema

        # --- 3. Volatility Filter (ATR) ---
        atr_series = _calculate_atr(pair_data.warm, ATR_PERIOD)
        # Need at least 2 ATR values to calculate a meaningful mean and standard deviation.
        if len(atr_series) < 2:
            continue
        
        current_atr = atr_series[-1]
        
        # Calculate historical mean and standard deviation of ATR from the available series.
        historical_atr_mean = statistics.mean(atr_series)
        historical_atr_std = statistics.stdev(atr_series) if len(atr_series) > 1 else 0.0

        # Check if current ATR is within the stability band.
        # Special handling for when historical_atr_std is 0 (all ATR values are the same).
        if historical_atr_std == 0:
            is_volatility_stable = (current_atr == historical_atr_mean)
        else:
            lower_bound = historical_atr_mean - VOLATILITY_BAND_MULTIPLIER * historical_atr_std
            upper_bound = historical_atr_mean + VOLATILITY_BAND_MULTIPLIER * historical_atr_std
            is_volatility_stable = (current_atr > lower_bound) and (current_atr < upper_bound)

        # --- Signal Generation ---
        ts = pair_data.hot[-1].polled_at
        price = pair_data.hot[-1].last_price

        if is_volatility_stable:
            # BUY signal: Cycle trough AND Uptrend
            # cos_phase < -TROUGH_THRESHOLD indicates the cycle is at a trough (phase near pi for a cosine wave).
            # This is consistent with the original rule's interpretation for a BUY signal.
            if cos_phase < -TROUGH_THRESHOLD and is_uptrend:
                signals.append(BuySignal(pair=pair, timestamp=ts, price=price))
            # SELL signal: Cycle peak AND Downtrend
            # cos_phase > TROUGH_THRESHOLD indicates the cycle is at a peak (phase near 0 or 2pi for a cosine wave).
            # This is consistent with the original rule's interpretation for a SELL signal.
            elif cos_phase > TROUGH_THRESHOLD and is_downtrend:
                signals.append(SellSignal(pair=pair, timestamp=ts, price=price))

    return signals