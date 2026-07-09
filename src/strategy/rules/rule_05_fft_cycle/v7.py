from __future__ import annotations

import cmath
import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle


# --- Constants and Parameters ---
# Parameters from the original FFT rule (rule_05_fft_cycle_v1)
FFT_MIN_CANDLES = 12
AMPLITUDE_THRESHOLD = 0.3
TROUGH_THRESHOLD = 0.7

# New parameters for Volatility and Price Reversal Confirmation
SHORT_MA_PERIOD = 3  # Period for the short-term Moving Average
ATR_PERIOD = 14      # Period for Average True Range calculation
VOLATILITY_THRESHOLD_FACTOR = 0.005 # ATR must be > this fraction of current_close to signal
REVERSAL_CANDLE_WINDOW = 3 # Number of candles to consider for reversal patterns (simple patterns use last 2)

# Minimum candles required for all indicators to be calculated
# This ensures enough data for FFT, ATR (period + 1), SMA (period), and reversal patterns (at least 2)
REQUIRED_CANDLES = max(FFT_MIN_CANDLES, ATR_PERIOD + 1, SHORT_MA_PERIOD, REVERSAL_CANDLE_WINDOW)


# --- Helper Functions ---

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
    """Naive O(n²) DFT — correct and fast enough for the warm-tier window."""
    n = len(series)
    return [
        sum(series[t] * cmath.exp(-2j * math.pi * k * t / n) for t in range(n))
        for k in range(n)
    ]


def _calculate_atr(warm_candles: list[WarmCandle], period: int) -> float:
    """Calculates the Average True Range (ATR) for the given warm candles."""
    # Need at least `period + 1` candles to calculate the first true range and then average
    if len(warm_candles) < period + 1:
        return 0.0

    true_ranges = []
    # Start from the second candle to calculate True Range, as it needs a previous close
    for i in range(1, len(warm_candles)):
        current_candle = warm_candles[i]
        previous_candle = warm_candles[i-1]

        tr1 = current_candle.high - current_candle.low
        tr2 = abs(current_candle.high - previous_candle.close)
        tr3 = abs(current_candle.low - previous_candle.close)
        true_ranges.append(max(tr1, tr2, tr3))

    # Calculate SMA of the last 'period' true_ranges
    if len(true_ranges) < period:
        # This case should ideally be caught by the initial len(warm_candles) check
        return 0.0

    return statistics.mean(true_ranges[-period:])


def _calculate_sma(prices: list[float], period: int) -> float:
    """Calculates the Simple Moving Average (SMA) for the given prices."""
    if len(prices) < period:
        return 0.0
    return statistics.mean(prices[-period:])


def _is_bullish_reversal_pattern(warm_candles: list[WarmCandle], window: int) -> bool:
    """
    Checks for a simple bullish reversal pattern.
    A bullish reversal is indicated by the most recent candle being bullish (close > open)
    and closing higher than the previous candle's close, suggesting upward momentum.
    The 'window' parameter ensures enough data for context, though the pattern itself uses the last two.
    """
    if len(warm_candles) < 2: # Need at least two candles to compare
        return False

    current_candle = warm_candles[-1]
    previous_candle = warm_candles[-2]

    # Current candle is bullish (closed higher than it opened)
    is_current_candle_bullish = current_candle.close > current_candle.open
    # Current candle closed higher than the previous candle's close
    is_higher_close = current_candle.close > previous_candle.close

    return is_current_candle_bullish and is_higher_close


def _is_bearish_reversal_pattern(warm_candles: list[WarmCandle], window: int) -> bool:
    """
    Checks for a simple bearish reversal pattern.
    A bearish reversal is indicated by the most recent candle being bearish (close < open)
    and closing lower than the previous candle's close, suggesting downward momentum.
    The 'window' parameter ensures enough data for context, though the pattern itself uses the last two.
    """
    if len(warm_candles) < 2: # Need at least two candles to compare
        return False

    current_candle = warm_candles[-1]
    previous_candle = warm_candles[-2]

    # Current candle is bearish (closed lower than it opened)
    is_current_candle_bearish = current_candle.close < current_candle.open
    # Current candle closed lower than the previous candle's close
    is_lower_close = current_candle.close < previous_candle.close

    return is_current_candle_bearish and is_lower_close


# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'FFT Cycle with Volatility and Price Reversal Confirmation' trading rule.

    This rule enhances FFT cycle detection by adding a volatility filter and
    a short-term price reversal confirmation. It emits a Buy signal when the
    dominant FFT cycle indicates a trough, price has shown a short-term upward
    reversal, and volatility (ATR) is above a certain threshold.
    Conversely, a Sell signal is generated for a cycle peak with a confirmed
    downward reversal and sufficient volatility.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm data for all calculations and hot data for signal timestamp/price
        if len(pair_data.warm) < REQUIRED_CANDLES or not pair_data.hot:
            continue

        # Extract necessary data for calculations
        warm_closes = [c.close for c in pair_data.warm]
        current_close = warm_closes[-1]
        previous_close = warm_closes[-2] # Safe due to REQUIRED_CANDLES >= 2

        # Get latest tick data for signal generation
        ts = pair_data.hot[-1].polled_at
        price = pair_data.hot[-1].last_price

        # 1. Volatility Filter: Check if ATR is above a certain threshold relative to current price
        atr_value = _calculate_atr(pair_data.warm, ATR_PERIOD)
        volatility_threshold = current_close * VOLATILITY_THRESHOLD_FACTOR

        if atr_value <= volatility_threshold:
            continue  # No signal if volatility is too low

        # Calculate FFT dominant cycle phase
        detrended = _detrend(warm_closes)
        n = len(detrended)

        # Check for sufficient data for DFT after detrending (should be same as warm_closes)
        if n < FFT_MIN_CANDLES:
            continue

        X = _dft(detrended)

        # Find the dominant frequency component (excluding DC component k=0)
        # We only care about the first half of the spectrum for unique frequencies
        k_star = max(range(1, n // 2 + 1), key=lambda k: abs(X[k]))
        amplitude = 2 * abs(X[k_star]) / n

        # Filter out dominant cycles with low amplitude relative to price volatility
        price_std = statistics.stdev(warm_closes)
        if price_std == 0 or amplitude / price_std < AMPLITUDE_THRESHOLD:
            continue

        # Calculate the phase of the dominant cycle at the last data point
        # The phase is adjusted to reflect the end of the series (n-1)
        phase = 2 * math.pi * k_star * (n - 1) / n + cmath.phase(X[k_star])
        cos_phase = math.cos(phase)

        # Calculate Short-term Moving Average for reversal confirmation
        short_ma = _calculate_sma(warm_closes, SHORT_MA_PERIOD)

        # 2. FFT Cycle Trough Detection and Buy Confirmation
        if cos_phase < -TROUGH_THRESHOLD:  # Cycle indicates a trough
            # Price Reversal Confirmation for Buy
            # Option 1: Short MA crossover (current close above MA, previous close below or equal)
            ma_crossover_buy = (current_close > short_ma and previous_close <= short_ma)
            # Option 2: Bullish candlestick pattern
            bullish_pattern = _is_bullish_reversal_pattern(pair_data.warm, REVERSAL_CANDLE_WINDOW)

            if ma_crossover_buy or bullish_pattern:
                signals.append(BuySignal(pair=pair, timestamp=ts, price=price))

        # 3. FFT Cycle Peak Detection and Sell Confirmation
        elif cos_phase > TROUGH_THRESHOLD:  # Cycle indicates a peak
            # Price Reversal Confirmation for Sell
            # Option 1: Short MA crossover (current close below MA, previous close above or equal)
            ma_crossover_sell = (current_close < short_ma and previous_close >= short_ma)
            # Option 2: Bearish candlestick pattern
            bearish_pattern = _is_bearish_reversal_pattern(pair_data.warm, REVERSAL_CANDLE_WINDOW)

            if ma_crossover_sell or bearish_pattern:
                signals.append(SellSignal(pair=pair, timestamp=ts, price=price))

    return signals