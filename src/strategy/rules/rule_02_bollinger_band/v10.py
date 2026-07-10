from __future__ import annotations

import statistics
import numpy as np
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick

# Parameters
BB_PERIOD: int = 20
BB_STD_DEV: float = 2.0
RSI_PERIOD: int = 14
ADAPTIVE_RSI_WINDOW: int = 60
RSI_OVERSOLD_PERCENTILE: float = 10.0  # Percentiles are typically float (0-100)
RSI_OVERBOUGHT_PERCENTILE: float = 90.0

# Minimum data for calculations:
# 1. Bollinger Bands require BB_PERIOD closes.
# 2. To calculate one RSI value, RSI_PERIOD + 1 closes are needed.
# 3. To get ADAPTIVE_RSI_WINDOW historical RSI values for percentile calculation,
#    we need ADAPTIVE_RSI_WINDOW + RSI_PERIOD closes in total.
# We take the maximum of these requirements to ensure enough data for all indicators.
MIN_WARM_CANDLES_REQUIRED: int = max(BB_PERIOD, ADAPTIVE_RSI_WINDOW + RSI_PERIOD)


def _calculate_rsi_series(closes: list[float], period: int) -> list[float]:
    """
    Calculates a series of RSI values using Wilder's smoothing method.
    The returned list contains RSI values, where the last value corresponds
    to the last close in the input series.
    The length of the returned list will be `len(closes) - period`.
    """
    if len(closes) < period + 1:
        return []

    rsi_values = []

    # Use numpy for efficient array operations on initial deltas
    np_closes = np.array(closes)
    deltas = np.diff(np_closes)  # deltas has length len(closes) - 1

    # Calculate initial average gain/loss over the first 'period' deltas.
    # These deltas correspond to closes[1]...closes[period+1].
    initial_gains = np.maximum(0, deltas[:period])
    initial_losses = np.abs(np.minimum(0, deltas[:period]))

    avg_gain = np.mean(initial_gains)
    avg_loss = np.mean(initial_losses)

    # Calculate the first RSI value (corresponds to closes[period])
    if avg_loss == 0:
        # Handle division by zero for RS; if no losses, RSI is 100 (if gains) or 50 (if no change)
        initial_rsi = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        initial_rsi = 100 - (100 / (1 + rs))
    rsi_values.append(initial_rsi)

    # Subsequent calculations using Wilder's smoothing
    # We iterate from the delta that corresponds to closes[period+1], which is deltas[period].
    for i in range(period, len(deltas)):
        current_delta = deltas[i]

        current_gain = max(0.0, current_delta)
        current_loss = abs(min(0.0, current_delta))

        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period

        if avg_loss == 0:
            rsi_values.append(100.0 if avg_gain > 0 else 50.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

    return rsi_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm candle data is available for all calculations
        if len(pair_data.warm) < MIN_WARM_CANDLES_REQUIRED:
            continue

        # Extract close prices from warm candles
        closes = [c.close for c in pair_data.warm]

        # --- Calculate Bollinger Bands ---
        # We use the last BB_PERIOD closes for SMA and STDDEV
        bb_closes = closes[-BB_PERIOD:]

        mid_band = statistics.mean(bb_closes)

        # Handle zero standard deviation to prevent errors and meaningless bands
        if len(bb_closes) < 2:  # stdev requires at least 2 points
            std_dev = 0.0
        else:
            std_dev = statistics.stdev(bb_closes)

        if std_dev == 0:  # Flat market, bands collapse, no meaningful signal
            continue

        upper_band = mid_band + (std_dev * BB_STD_DEV)
        lower_band = mid_band - (std_dev * BB_STD_DEV)

        # --- Calculate RSI ---
        # Get the series of RSI values. The last value is the current RSI.
        all_rsi_values = _calculate_rsi_series(closes, RSI_PERIOD)
        if not all_rsi_values:  # Not enough data to calculate any RSI values
            continue

        rsi_value = all_rsi_values[-1]  # The most recent RSI value

        # --- Calculate Adaptive RSI thresholds ---
        # We need the last ADAPTIVE_RSI_WINDOW of the calculated RSI values
        if len(all_rsi_values) < ADAPTIVE_RSI_WINDOW:
            # This should ideally be caught by MIN_WARM_CANDLES_REQUIRED,
            # but serves as an extra safeguard.
            continue

        recent_rsi_history = all_rsi_values[-ADAPTIVE_RSI_WINDOW:]

        # Ensure there's enough data for percentile calculation.
        # np.percentile can work with 1 element, but a distribution is more meaningful.
        if len(recent_rsi_history) < 2:
            continue

        adaptive_rsi_oversold_threshold = np.percentile(recent_rsi_history, RSI_OVERSOLD_PERCENTILE)
        adaptive_rsi_overbought_threshold = np.percentile(recent_rsi_history, RSI_OVERBOUGHT_PERCENTILE)

        # --- Generate signals ---
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # Buy signal: Price drops below lower BB AND RSI is below its adaptive oversold threshold
        if current_price < lower_band and rsi_value < adaptive_rsi_oversold_threshold:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Sell signal: Price rises above upper BB AND RSI is above its adaptive overbought threshold
        elif current_price > upper_band and rsi_value > adaptive_rsi_overbought_threshold:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals