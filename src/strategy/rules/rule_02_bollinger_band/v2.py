from __future__ import annotations

import statistics
import numpy as np

from src.agent.models import BuySignal, MarketData, SellSignal

# --- Constants ---
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70

# Minimum candles needed for both BB (BB_PERIOD) and RSI (RSI_PERIOD + 1 for initial calculation)
MIN_CANDLES = max(BB_PERIOD, RSI_PERIOD + 1)


# --- Helper Functions ---

def calculate_rsi(prices: list[float], period: int) -> float | None:
    """
    Calculates the Relative Strength Index (RSI) for a given list of prices.
    Requires at least `period + 1` prices to calculate the first RSI value.
    """
    if len(prices) < period + 1:
        return None

    prices_arr = np.array(prices, dtype=float)
    price_diffs = np.diff(prices_arr)

    # Separate gains and losses
    up = np.where(price_diffs > 0, price_diffs, 0)
    down = np.where(price_diffs < 0, -price_diffs, 0)

    # Initialize arrays for average gains and losses (smoothed moving average)
    avg_up = np.zeros_like(up)
    avg_down = np.zeros_like(down)

    # Calculate the first average gain and loss using a simple moving average
    avg_up[period - 1] = np.mean(up[:period])
    avg_down[period - 1] = np.mean(down[:period])

    # Calculate subsequent averages using the Wilder's smoothing method (RMA)
    for i in range(period, len(up)):
        avg_up[i] = (avg_up[i - 1] * (period - 1) + up[i]) / period
        avg_down[i] = (avg_down[i - 1] * (period - 1) + down[i]) / period

    # Calculate Relative Strength (RS)
    # Handle division by zero if avg_down is zero
    rs = np.where(avg_down == 0, np.inf, avg_up / avg_down)

    # Calculate RSI
    # If RS is infinity (no losses), RSI is 100.
    rsi_values = np.where(rs == np.inf, 100, 100 - (100 / (1 + rs)))

    # Return the latest RSI value
    return float(rsi_values[-1])


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on Bollinger Bands with RSI confirmation.

    A Buy signal is generated if the current price falls below the lower Bollinger Band
    AND RSI indicates an oversold condition (RSI < RSI_OVERSOLD_THRESHOLD).
    A Sell signal is generated if the current price rises above the upper Bollinger Band
    AND RSI indicates an overbought condition (RSI > RSI_OVERBOUGHT_THRESHOLD).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for calculations and at least one hot tick
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES:
            continue

        # Extract close prices from warm candles (historical closed prices)
        closes = [c.close for c in pair_data.warm]

        # --- Calculate Bollinger Bands ---
        # Use the last BB_PERIOD closes for SMA and StdDev
        bb_closes = closes[-BB_PERIOD:]
        mean = statistics.mean(bb_closes)
        std = statistics.stdev(bb_closes)

        # If standard deviation is zero, bands collapse, skip to avoid division by zero
        # or meaningless signals.
        if std == 0:
            continue

        upper_band = mean + BB_STD_DEV_MULTIPLIER * std
        lower_band = mean - BB_STD_DEV_MULTIPLIER * std

        # --- Calculate RSI ---
        # Use the last RSI_PERIOD + 1 closes for RSI calculation
        rsi_closes = closes[-(RSI_PERIOD + 1):]
        rsi_value = calculate_rsi(rsi_closes, RSI_PERIOD)

        # If RSI cannot be calculated due to insufficient data (should be caught by MIN_CANDLES, but defensive)
        if rsi_value is None:
            continue

        # --- Signal Generation ---
        # Get the current price from the latest hot tick
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        if current_price < lower_band and rsi_value < RSI_OVERSOLD_THRESHOLD:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif current_price > upper_band and rsi_value > RSI_OVERBOUGHT_THRESHOLD:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals