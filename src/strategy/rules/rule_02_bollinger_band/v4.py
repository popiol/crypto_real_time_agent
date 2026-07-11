"""Rule 02 — Bollinger Band with Dynamic RSI Thresholds (v4)."""
from __future__ import annotations

import numpy as np
from datetime import datetime

from src.agent.models import BuySignal, MarketData, SellSignal, Tick, WarmCandle, PairData

# --- Constants ---
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0
RSI_PERIOD = 14
RSI_MEAN_STD_PERIOD = 30  # Period for calculating mean and stddev of RSI
RSI_STD_DEV_MULTIPLIER_K = 1.5  # Multiplier for RSI standard deviation

# Minimum candles needed for Bollinger Bands (BB_PERIOD) and for the RSI series.
# To calculate `RSI_MEAN_STD_PERIOD` RSI values, we need `RSI_PERIOD + RSI_MEAN_STD_PERIOD` prices.
# Example: If RSI_PERIOD=14, RSI_MEAN_STD_PERIOD=30:
# We need 14+1 prices for the first RSI value, and then 29 more prices for the subsequent 29 RSI values.
# Total prices = (14+1) + (30-1) = 15 + 29 = 44 prices.
# This simplifies to RSI_PERIOD + RSI_MEAN_STD_PERIOD prices.
RSI_MIN_PRICES_FOR_SERIES = RSI_PERIOD + RSI_MEAN_STD_PERIOD
MIN_CANDLES = max(BB_PERIOD, RSI_MIN_PRICES_FOR_SERIES)


# --- Helper Functions ---

def calculate_rsi_series(prices: list[float], period: int) -> np.ndarray | None:
    """
    Calculates the Relative Strength Index (RSI) for a given list of prices,
    returning a series of valid RSI values.
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
    # These arrays will store the average gains/losses for each point where RSI can be calculated.
    avg_up = np.zeros_like(up)
    avg_down = np.zeros_like(down)

    # Calculate the first average gain and loss using a simple moving average over the first `period` diffs
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

    # Return only the valid RSI values, which start from the (period-1)-th index.
    # The first 'period' values are used to establish the initial average.
    return rsi_values[period - 1:]


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on Bollinger Bands with dynamically adjusted RSI thresholds.

    A Buy signal is generated if the current price falls below the lower Bollinger Band
    AND RSI is below a dynamically calculated oversold threshold.
    A Sell signal is generated if the current price rises above the upper Bollinger Band
    AND RSI is above a dynamically calculated overbought threshold.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for all calculations and at least one hot tick
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES:
            continue

        # Extract close prices from warm candles (historical closed prices)
        closes = [c.close for c in pair_data.warm]

        # --- Calculate Bollinger Bands ---
        # Bollinger Bands are typically calculated on the latest `BB_PERIOD` closing prices.
        # We need at least BB_PERIOD candles for this.
        if len(closes) < BB_PERIOD:  # Defensive check, should be covered by MIN_CANDLES
            continue

        bb_closes = np.array(closes[-BB_PERIOD:], dtype=float)
        mean_bb = np.mean(bb_closes)
        std_bb = np.std(bb_closes)

        # If standard deviation is zero, bands collapse, skip to avoid meaningless signals.
        if std_bb == 0:
            continue

        upper_band = mean_bb + BB_STD_DEV_MULTIPLIER * std_bb
        lower_band = mean_bb - BB_STD_DEV_MULTIPLIER * std_bb

        # --- Calculate RSI Series ---
        # We need enough prices to generate `RSI_MEAN_STD_PERIOD` RSI values.
        # This is `RSI_MIN_PRICES_FOR_SERIES`, which is covered by `MIN_CANDLES`.
        rsi_series_closes = closes[-RSI_MIN_PRICES_FOR_SERIES:]
        all_rsi_values = calculate_rsi_series(rsi_series_closes, RSI_PERIOD)

        # Ensure we have enough RSI values to calculate dynamic thresholds
        if all_rsi_values is None or len(all_rsi_values) < RSI_MEAN_STD_PERIOD:
            continue

        # --- Calculate Dynamic RSI Thresholds ---
        current_rsi = all_rsi_values[-1]  # The latest RSI value

        # Take the last `RSI_MEAN_STD_PERIOD` RSI values to calculate their mean and stddev
        dynamic_rsi_window = all_rsi_values[-RSI_MEAN_STD_PERIOD:]

        rsi_mean = np.mean(dynamic_rsi_window)
        rsi_std = np.std(dynamic_rsi_window)

        # If RSI standard deviation is zero, the dynamic thresholds will simply be the RSI mean.
        # This is still a valid (though simplified) dynamic threshold, so we don't need to skip.
        dynamic_oversold_rsi_threshold = rsi_mean - RSI_STD_DEV_MULTIPLIER_K * rsi_std
        dynamic_overbought_rsi_threshold = rsi_mean + RSI_STD_DEV_MULTIPLIER_K * rsi_std

        # --- Signal Generation ---
        # Get the current price and timestamp from the latest hot tick
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        if current_price < lower_band and current_rsi < dynamic_oversold_rsi_threshold:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif current_price > upper_band and current_rsi > dynamic_overbought_rsi_threshold:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals