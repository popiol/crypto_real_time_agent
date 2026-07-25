from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime
from pydantic import Field

# Rule specific constants
RULE_ID = "RSI_SMA_Relaxed_Timing_V1"
RSI_PERIOD = 14
SHORT_SMA_PERIOD = 10  # From pseudocode
LONG_SMA_PERIOD = 30   # From pseudocode
BUY_RSI_THRESHOLD = 35 # From pseudocode
SELL_RSI_THRESHOLD = 65 # From pseudocode
WINDOW_SIZE = 3 # From pseudocode (number of latest hourly candles to check for confluence)

# Minimum data required for calculations.
# RSI(14) needs (period + 2) candles for current and previous RSI for crossover detection. (14+2=16)
# SMA(10) needs (period + 1) candles for current and previous SMA for crossover detection. (10+1=11)
# SMA(30) needs (period + 1) candles for current and previous SMA for crossover detection. (30+1=31)
#
# The maximum of these base requirements is 31 (for LONG_SMA_PERIOD).
# We need to check conditions within the last `WINDOW_SIZE` candles.
# For the `i`-th iteration of the window loop (where `i` is 0 to `WINDOW_SIZE-1`),
# we are examining the candle at `warm_candles[-(i+1)]` (current) and `warm_candles[-(i+2)]` (previous).
# The slice of `close_prices` passed to `_calculate_rsi` and `_calculate_sma` will be
# `close_prices[:len(close_prices) - i]`.
# For the deepest lookback, when `i = WINDOW_SIZE - 1`, the length of this slice is
# `len(close_prices) - (WINDOW_SIZE - 1)`.
# This slice must be long enough for the indicator calculations:
# `len(close_prices) - (WINDOW_SIZE - 1) >= max(RSI_PERIOD + 2, LONG_SMA_PERIOD + 1)`
# `len(close_prices) - (3 - 1) >= max(14 + 2, 30 + 1)`
# `len(close_prices) - 2 >= max(16, 31)`
# `len(close_prices) - 2 >= 31`
# `len(close_prices) >= 31 + 2 = 33`.
MIN_CANDLES_REQUIRED = LONG_SMA_PERIOD + WINDOW_SIZE


def _calculate_rsi(prices: np.ndarray, period: int) -> tuple[float | None, float | None]:
    """
    Calculates the current and previous RSI values for the given prices.
    Returns (current_rsi, previous_rsi).
    Requires at least `period + 2` prices to calculate both current and previous RSI values.
    """
    if len(prices) < period + 2:
        return None, None

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsis = np.full(len(prices), np.nan)

    # Initial average gain/loss for the first `period` deltas
    # (i.e., corresponding to prices[0]...prices[period])
    if len(gains) < period:
        return None, None

    avg_gain_prev = np.mean(gains[:period])
    avg_loss_prev = np.mean(losses[:period])

    # Calculate first RSI value (for prices[period])
    if avg_loss_prev == 0:
        rs = np.inf
    else:
        rs = avg_gain_prev / avg_loss_prev
    rsis[period] = 100 - (100 / (1 + rs)) if not np.isinf(rs) else 100

    # Exponential smoothing for subsequent periods
    # Loop starts from the price index `period + 1` up to the last price.
    for i in range(period + 1, len(prices)):
        current_gain = gains[i - 1]
        current_loss = losses[i - 1]

        avg_gain_curr = ((avg_gain_prev * (period - 1)) + current_gain) / period
        avg_loss_curr = ((avg_loss_prev * (period - 1)) + current_loss) / period

        if avg_loss_curr == 0:
            rs = np.inf
        else:
            rs = avg_gain_curr / avg_loss_curr

        rsis[i] = 100 - (100 / (1 + rs)) if not np.isinf(rs) else 100

        avg_gain_prev = avg_gain_curr
        avg_loss_prev = avg_loss_curr

    if not np.isnan(rsis[-1]) and not np.isnan(rsis[-2]):
        return float(rsis[-1]), float(rsis[-2])
    return None, None


def _calculate_sma(prices: np.ndarray, period: int) -> tuple[float | None, float | None]:
    """
    Calculates the current and previous SMA values for the given prices.
    Returns (current_sma, previous_sma).
    Requires at least `period + 1` prices to calculate both current and previous SMA values.
    """
    if len(prices) < period + 1:
        return None, None

    current_sma = np.mean(prices[-period:])
    previous_sma = np.mean(prices[-(period + 1):-1])

    return float(current_sma), float(previous_sma)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Constraint: data.warm holds at most 24 hourly candles.
        # This rule requires MIN_CANDLES_REQUIRED = 33 candles.
        # Therefore, this rule will never generate signals under the given data constraints.
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        close_prices = np.array([c.close for c in warm_candles])

        # Get the latest timestamp and price for signal generation
        latest_candle = warm_candles[-1]
        latest_timestamp = latest_candle.hour
        latest_price = latest_candle.close

        # Flags to track if conditions occurred within the WINDOW_SIZE
        sma_crossover_up_in_window = False
        rsi_cross_down_in_window = False
        sma_crossover_down_in_window = False
        rsi_cross_up_in_window = False

        # Loop through the last WINDOW_SIZE candles (0-indexed from latest)
        # i=0 refers to the latest candle, i=1 to the second latest, etc.
        for i in range(WINDOW_SIZE):
            # Create a slice of prices to include up to the (i+1)-th candle from the end.
            # E.g., for i=0, `current_slice` includes all `close_prices` (for latest candle).
            # E.g., for i=1, `current_slice` includes all but the latest candle (for second latest).
            # E.g., for i=2, `current_slice` includes all but the two latest candles (for third latest).
            # The `_calculate_` functions will then operate on the last elements of this slice.
            current_slice = close_prices[:len(close_prices) - i]

            # Calculate indicators for the current candle (last in slice) and its predecessor
            current_rsi, prev_rsi = _calculate_rsi(current_slice, RSI_PERIOD)
            current_sma_short, prev_sma_short = _calculate_sma(current_slice, SHORT_SMA_PERIOD)
            current_sma_long, prev_sma_long = _calculate_sma(current_slice, LONG_SMA_PERIOD)

            # If any indicator cannot be calculated for this window position, skip this iteration.
            # This handles cases where `current_slice` is too short for a specific indicator,
            # which could happen if MIN_CANDLES_REQUIRED was not perfectly tuned or data is malformed.
            if any(val is None for val in [current_rsi, prev_rsi, current_sma_short, prev_sma_short, current_sma_long, prev_sma_long]):
                continue

            # Check for SMA crossover up (potential buy condition)
            if current_sma_short > current_sma_long and prev_sma_short <= prev_sma_long:
                sma_crossover_up_in_window = True

            # Check for RSI cross below BUY_RSI_THRESHOLD (potential buy condition)
            if current_rsi < BUY_RSI_THRESHOLD and prev_rsi >= BUY_RSI_THRESHOLD:
                rsi_cross_down_in_window = True

            # Check for SMA crossover down (potential sell condition)
            if current_sma_short < current_sma_long and prev_sma_short >= prev_sma_long:
                sma_crossover_down_in_window = True

            # Check for RSI cross above SELL_RSI_THRESHOLD (potential sell condition)
            if current_rsi > SELL_RSI_THRESHOLD and prev_rsi <= SELL_RSI_THRESHOLD:
                rsi_cross_up_in_window = True

        # --- Generate Signals based on confluence within the window ---
        # A buy signal requires both an upward SMA crossover AND an RSI cross below threshold
        # to have occurred at any point within the WINDOW_SIZE.
        if sma_crossover_up_in_window and rsi_cross_down_in_window:
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_timestamp,
                price=latest_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))
        # A sell signal requires both a downward SMA crossover AND an RSI cross above threshold
        # to have occurred at any point within the WINDOW_SIZE.
        elif sma_crossover_down_in_window and rsi_cross_up_in_window:
            signals.append(SellSignal(
                pair=pair,
                timestamp=latest_timestamp,
                price=latest_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))

    return signals