from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal

# Rule parameters
FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9

# Minimum data points required for MACD and Signal Line calculation.
# For the `_calculate_ema` function, which initializes the first EMA value
# with the first price and then applies the formula iteratively,
# we need at least `SLOW_PERIOD` data points for the `ema_slow` to be calculated
# for all points in the series.
# The resulting `macd_line` will then have `len(prices)` points.
# The `signal_line` is then an EMA of `macd_line`. For it to be calculated with
# sufficient points for crossover detection (at least two points),
# `len(macd_line)` (which is `len(prices)`) must be at least `SIGNAL_PERIOD`.
# Therefore, `len(prices)` must be at least `max(SLOW_PERIOD, SIGNAL_PERIOD)`.
# This ensures that `ema_fast`, `ema_slow`, `macd_line`, and `signal_line`
# all have at least enough elements for `[-1]` and `[-2]` access to detect a crossover.
MIN_TICKS_REQUIRED = max(FAST_PERIOD, SLOW_PERIOD, SIGNAL_PERIOD)


def _calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates the Exponential Moving Average (EMA) for a given price series.
    The EMA is initialized with the first price in the series, and then the
    standard EMA formula is applied.
    Returns an array of the same length as the input `prices`.
    """
    if len(prices) == 0:
        return np.array([])

    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices, dtype=float)

    ema[0] = prices[0]  # Initialize with the first price

    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot

        # Ensure we have enough data points for MACD calculation.
        if len(ticks) < MIN_TICKS_REQUIRED:
            continue

        # Extract 'last_price' values from the ticks
        prices = np.array([t.last_price for t in ticks])

        # Calculate EMAs for the MACD Line
        ema_fast = _calculate_ema(prices, FAST_PERIOD)
        ema_slow = _calculate_ema(prices, SLOW_PERIOD)

        # Calculate the MACD Line
        macd_line = ema_fast - ema_slow

        # Calculate the Signal Line (EMA of the MACD Line)
        signal_line = _calculate_ema(macd_line, SIGNAL_PERIOD)

        # Ensure there are at least two data points for crossover detection.
        # This check is largely redundant if MIN_TICKS_REQUIRED is correctly set,
        # but provides an extra layer of robustness.
        if len(macd_line) < 2 or len(signal_line) < 2:
            continue

        # Get the latest and previous MACD and Signal Line values
        current_macd = macd_line[-1]
        prev_macd = macd_line[-2]
        current_signal = signal_line[-1]
        prev_signal = signal_line[-2]

        # Detect crossovers with Zero-Line Confirmation
        # Buy signal: MACD line crosses above Signal line AND MACD line is above zero
        # The condition "MACD_Line > 0 (or MACD_Line crosses above 0 shortly after)"
        # simplifies to `current_macd > 0` as any recent cross above zero
        # would imply `current_macd` is currently positive.
        if (prev_macd <= prev_signal and current_macd > current_signal) and current_macd > 0:
            signals.append(BuySignal(
                pair=pair,
                timestamp=ticks[-1].polled_at,
                price=ticks[-1].last_price,
                rule_id="da9b4306-ab6c-4a19-917a-38f81d73d63a", # Using idea_id as rule_id
                confidence=None
            ))
        # Sell signal: MACD line crosses below Signal line AND MACD line is below zero
        # Similarly, "MACD_Line < 0 (or MACD_Line crosses below 0 shortly after)"
        # simplifies to `current_macd < 0`.
        elif (prev_macd >= prev_signal and current_macd < current_signal) and current_macd < 0:
            signals.append(SellSignal(
                pair=pair,
                timestamp=ticks[-1].polled_at,
                price=ticks[-1].last_price,
                rule_id="da9b4306-ab6c-4a19-917a-38f81d73d63a", # Using idea_id as rule_id
                confidence=None
            ))

    return signals