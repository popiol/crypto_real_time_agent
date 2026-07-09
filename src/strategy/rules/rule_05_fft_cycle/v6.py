from __future__ import annotations

import cmath
import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# FFT Cycle Parameters (from original rule)
MIN_CANDLES = 12  # Minimum warm candles for FFT analysis
AMPLITUDE_THRESHOLD = 0.3  # Peak amplitude of the dominant cycle must be at least this fraction of price std
TROUGH_THRESHOLD = 0.7  # cos(phase) must be below −TROUGH_THRESHOLD for a trough, above for a peak

# RSI Confirmation Parameters (new)
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Combined minimum data requirement for both FFT and RSI
# For FFT, MIN_CANDLES is 12.
# For RSI, we need at least RSI_PERIOD + 2 candles to get two RSI values (current and previous)
# 14 + 2 = 16 candles.
# So, the overall minimum is max(12, 16) = 16 candles.
MIN_TOTAL_CANDLES = max(MIN_CANDLES, RSI_PERIOD + 2)


def _detrend(series: list[float]) -> list[float]:
    """Remove the least-squares linear trend so the DFT reflects cycles, not drift."""
    n = len(series)
    if n < 2:
        return series
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


def _calculate_rsi(prices: list[float], period: int) -> list[float]:
    """
    Calculates the Relative Strength Index (RSI) for a given price series.
    Returns a list of RSI values, where the last element is the most recent.
    Requires at least `period + 1` prices to calculate the first RSI value,
    and `period + 2` prices to calculate at least two RSI values needed for turn-up/down.
    """
    if len(prices) <= period:
        return []

    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

    gains = [c if c > 0 else 0.0 for c in changes]
    losses = [abs(c) if c < 0 else 0.0 for c in changes]

    rsi_values = []

    # Calculate initial average gain and loss over the first 'period' changes
    initial_gains = gains[:period]
    initial_losses = losses[:period]
    
    # Handle cases where initial_gains or initial_losses might be empty due to period=0 or small data.
    # The check len(prices) <= period already handles this for period > 0.
    # If period is 0, this would be an issue, but RSI period is typically >= 1.
    if not initial_gains: # Should not happen with period > 0 and len(prices) > period
        return []

    avg_gain = sum(initial_gains) / period
    avg_loss = sum(initial_losses) / period

    # Calculate initial RSI
    if avg_loss == 0:
        rsi = 100.0 if avg_gain > 0 else 50.0 # If no losses, RSI is 100. If no changes, 50.
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    rsi_values.append(rsi)

    # Calculate subsequent RSI values using smoothing
    for i in range(period, len(gains)):
        current_gain = gains[i]
        current_loss = losses[i]

        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period

        if avg_loss == 0:
            rsi = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)

    return rsi_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm candle data for both FFT and RSI
        if len(pair_data.warm) < MIN_TOTAL_CANDLES or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        n = len(prices)

        # 1. FFT Cycle Detection
        detrended = _detrend(prices)
        
        # If detrended series is too short after processing (e.g., n < 2 for detrend), skip
        if len(detrended) < MIN_CANDLES:
            continue

        X = _dft(detrended)
        if not X: # dft might return empty if input is empty
            continue

        # Find dominant cycle (excluding DC component k=0)
        # We need at least 2 points for k_star to be meaningful (n//2+1 >= 1)
        # If n=1, n//2+1 = 1, range(1,1) is empty.
        # If n=2, n//2+1 = 2, range(1,2) is [1]. k_star = 1.
        if n < 2:
            continue # Already handled by MIN_TOTAL_CANDLES, but defensive.

        k_star_range = range(1, n // 2 + 1)
        if not k_star_range: # No meaningful cycle can be found
            continue
        
        k_star = max(k_star_range, key=lambda k: abs(X[k]))
        amplitude = 2 * abs(X[k_star]) / n

        # Check amplitude significance
        try:
            price_std = statistics.stdev(prices)
        except statistics.StatisticsError: # Handles case of all same prices, stdev is 0
            price_std = 0.0
        
        if price_std == 0 or amplitude / price_std < AMPLITUDE_THRESHOLD:
            continue

        # Calculate dominant cycle phase
        phase = 2 * math.pi * k_star * (n - 1) / n + cmath.phase(X[k_star])
        cos_phase = math.cos(phase)

        # 2. RSI Calculation
        rsi_values = _calculate_rsi(prices, RSI_PERIOD)
        
        # Ensure we have at least two RSI values for turn-up/down check
        if len(rsi_values) < 2:
            continue
        
        current_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2]

        # Get latest market data for signal
        ts = pair_data.hot[-1].polled_at
        price = pair_data.hot[-1].last_price

        # 3. Generate Signals with Confirmation
        # Buy Signal: Cycle trough + Oversold RSI + RSI turn-up
        if cos_phase < -TROUGH_THRESHOLD:  # Cycle indicates a trough
            if current_rsi < RSI_OVERSOLD and current_rsi > prev_rsi:  # RSI oversold and turning up
                signals.append(BuySignal(pair=pair, timestamp=ts, price=price))
        
        # Sell Signal: Cycle peak + Overbought RSI + RSI turn-down
        elif cos_phase > TROUGH_THRESHOLD:  # Cycle indicates a peak
            if current_rsi > RSI_OVERBOUGHT and current_rsi < prev_rsi:  # RSI overbought and turning down
                signals.append(SellSignal(pair=pair, timestamp=ts, price=price))

    return signals