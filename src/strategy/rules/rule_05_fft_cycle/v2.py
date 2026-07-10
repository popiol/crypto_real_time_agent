from __future__ import annotations

import cmath
import math
import statistics

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


# Minimum warm candles; fewer than half a 24-hour window is not enough for cycle detection
MIN_CANDLES = 12

# Peak amplitude of the dominant cycle must be at least this fraction of price std
AMPLITUDE_THRESHOLD = 0.3

# cos(phase) must be below −TROUGH_THRESHOLD to be considered a trough
TROUGH_THRESHOLD = 0.7

# NEW: Constants for Volatility Adaptive Filtering and Trend Confirmation
ATR_WINDOW = 14  # Window for Average True Range
VOLATILITY_THRESHOLD_PERCENTILE = 80 # Percentile of historical ATRs for dynamic threshold
# Note: For `VOLATILITY_THRESHOLD_PERCENTILE`, we use the available ATRs within the warm window.
# With 24 candles and ATR_WINDOW=14, we can compute 24-14+1 = 11 ATR values.
# The 80th percentile will be based on these 11 values.

EMA_WINDOW = 12  # Window for Exponential Moving Average (medium-term trend)
# The pseudocode suggested 50, but the 'warm' data is limited to at most 24 hourly candles.
# Adjusted to fit within available 24 hourly candles, making it a "medium-term" trend
# relative to the available short history.

# Minimum candles required for all calculations (FFT, ATR, EMA).
# ATR requires `ATR_WINDOW + 1` candles (for previous close reference in True Range calculation).
# EMA requires `EMA_WINDOW` candles (for initial SMA calculation).
# FFT requires `MIN_CANDLES`.
MIN_CANDLES_FOR_FILTERS = max(MIN_CANDLES, ATR_WINDOW + 1, EMA_WINDOW)


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


def _calculate_atr(highs: list[float], lows: list[float], closes: list[float], window: int) -> list[float]:
    """Calculates Average True Range (ATR) for a given window."""
    if len(highs) < window + 1:
        return []

    true_ranges = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        true_ranges.append(tr)

    atr_values = []
    if len(true_ranges) >= window:
        # Calculate initial SMA for the first ATR value
        initial_atr = statistics.mean(true_ranges[:window])
        atr_values.append(initial_atr)

        # Calculate subsequent ATRs using EMA formula
        multiplier = 2 / (window + 1)
        for i in range(window, len(true_ranges)):
            current_atr = (true_ranges[i] - atr_values[-1]) * multiplier + atr_values[-1]
            atr_values.append(current_atr)

    return atr_values


def _calculate_ema(series: list[float], window: int) -> list[float]:
    """Calculates Exponential Moving Average (EMA) for a given window."""
    if len(series) < window:
        return []

    ema_values = []
    # Initial EMA: Simple Moving Average of the first 'window' periods
    initial_ema = statistics.mean(series[:window])
    ema_values.append(initial_ema)

    multiplier = 2 / (window + 1)
    for i in range(window, len(series)):
        current_ema = (series[i] - ema_values[-1]) * multiplier + ema_values[-1]
        ema_values.append(current_ema)

    return ema_values


def _calculate_dynamic_volatility_threshold(atrs: list[float], percentile: int) -> float:
    """Calculates a dynamic volatility threshold based on a percentile of historical ATRs."""
    if not atrs:
        return 0.0 # Return a default low value to filter everything if no ATRs

    sorted_atrs = sorted(atrs)
    # Calculate index for percentile. Ensure index is within bounds.
    index = math.ceil(len(sorted_atrs) * percentile / 100) - 1
    index = max(0, min(index, len(sorted_atrs) - 1))

    return sorted_atrs[index]


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm candles for all calculations
        if len(pair_data.warm) < MIN_CANDLES_FOR_FILTERS or not pair_data.hot:
            continue

        # Extract data for calculations
        prices = [c.close for c in pair_data.warm]
        highs = [c.high for c in pair_data.warm]
        lows = [c.low for c in pair_data.warm]
        closes = [c.close for c in pair_data.warm]
        n = len(prices)

        # --- FFT Cycle Logic (Existing) ---
        detrended = _detrend(prices)
        X = _dft(detrended)

        # Find dominant cycle (excluding DC component k=0 and Nyquist frequency if n is even)
        # k_star ranges from 1 to n // 2.
        k_star = max(range(1, n // 2 + 1), key=lambda k: abs(X[k]))
        amplitude = 2 * abs(X[k_star]) / n

        price_std = statistics.stdev(prices)
        if price_std == 0 or amplitude / price_std < AMPLITUDE_THRESHOLD:
            continue # Cycle not significant enough relative to price volatility

        phase = 2 * math.pi * k_star * (n - 1) / n + cmath.phase(X[k_star])
        cos_phase = math.cos(phase)

        ts = pair_data.hot[-1].polled_at
        price = pair_data.hot[-1].last_price

        buy_signal_raw = (cos_phase < -TROUGH_THRESHOLD)
        sell_signal_raw = (cos_phase > TROUGH_THRESHOLD)

        # --- NEW: Volatility Adaptive Filtering ---
        atrs = _calculate_atr(highs, lows, closes, ATR_WINDOW)
        if not atrs:
            continue # Not enough data for ATR calculation

        current_atr = atrs[-1] # Short-term volatility (last ATR value)
        
        # Calculate dynamic volatility threshold from the available ATR history.
        # If there are fewer than 2 ATR values, a percentile isn't meaningful.
        # In such cases, use a heuristic (e.g., 1.5 times the mean of available ATRs) as a fallback.
        if len(atrs) < 2:
             volatility_threshold = statistics.mean(atrs) * 1.5
        else:
            volatility_threshold = _calculate_dynamic_volatility_threshold(atrs, VOLATILITY_THRESHOLD_PERCENTILE)
        
        if current_atr >= volatility_threshold:
            continue # Filter out signals if market volatility exceeds the dynamic threshold

        # --- NEW: Trend Confirmation ---
        emas = _calculate_ema(closes, EMA_WINDOW)
        if len(emas) < 2: # Need at least two EMA values to determine slope
            continue # Not enough data for EMA trend determination

        medium_term_trend_up = (emas[-1] > emas[-2])
        medium_term_trend_down = (emas[-1] < emas[-2])

        # --- Modified Signal Conditions ---
        if buy_signal_raw:
            # Confirm buy signal with an upward medium-term trend
            if medium_term_trend_up:
                signals.append(BuySignal(pair=pair, timestamp=ts, price=price))
        elif sell_signal_raw:
            # Confirm sell signal with a downward medium-term trend
            if medium_term_trend_down:
                signals.append(SellSignal(pair=pair, timestamp=ts, price=price))

    return signals