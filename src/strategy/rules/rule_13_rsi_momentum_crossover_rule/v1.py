from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule Configuration ---
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 60
RSI_SELL_THRESHOLD = 40

# Minimum number of candles required for a valid signal:
# RSI_PERIOD + 1 candles are needed to calculate the first RSI value.
# RSI_PERIOD + 2 candles are needed to have two consecutive RSI values
# (current and previous) for a crossover comparison.
MIN_CANDLES_FOR_RSI = RSI_PERIOD + 2

# --- Helper Function for RSI Calculation ---
def _calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates the Relative Strength Index (RSI) for a given array of closing prices.
    The returned array will have NaN for the first 'period' elements,
    as RSI cannot be calculated without sufficient preceding data.

    Args:
        prices (np.ndarray): A 1D numpy array of closing prices.
        period (int): The lookback period for RSI calculation.

    Returns:
        np.ndarray: A 1D numpy array of RSI values, with NaNs for the initial 'period' elements.
    """
    if len(prices) < period + 1:
        # Not enough data for even one RSI value
        return np.full(len(prices), np.nan)

    # Calculate price changes (deltas)
    # `np.diff` returns an array of length `len(prices) - 1`
    delta = np.diff(prices)

    # Separate positive and negative changes
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0) # Convert negative changes to positive values

    # Initialize RSI array with NaNs. Its length should match the prices array.
    rsi_values = np.full(len(prices), np.nan)

    # Calculate initial average gains and losses for the first 'period' deltas.
    # These correspond to prices[1] through prices[period].
    avg_up = np.mean(up[:period])
    avg_down = np.mean(down[:period])

    # Calculate the first RSI value. This value corresponds to `prices[period]`.
    if avg_down == 0:
        rs = np.inf if avg_up > 0 else 0  # Handle cases where there are no losses
    else:
        rs = avg_up / avg_down
    
    rsi_values[period] = 100 - (100 / (1 + rs))

    # Calculate subsequent RSI values using the smoothing formula (Wilder's smoothing)
    # The loop starts from `prices[period + 1]`, which corresponds to `delta[period]`.
    for i in range(period + 1, len(prices)):
        # Update average gains and losses using the smoothing formula
        # `up[i-1]` and `down[i-1]` correspond to the delta from `prices[i-1]` to `prices[i]`
        avg_up = (avg_up * (period - 1) + up[i-1]) / period
        avg_down = (avg_down * (period - 1) + down[i-1]) / period

        if avg_down == 0:
            rs = np.inf if avg_up > 0 else 0
        else:
            rs = avg_up / avg_down
        
        rsi_values[i] = 100 - (100 / (1 + rs))

    return rsi_values

# --- Main Signal Generation Function ---
def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates Buy/Sell signals based on the RSI Momentum Crossover rule.

    A Buy signal is generated when the 14-period RSI crosses above a bullish threshold (60),
    indicating strengthening upward momentum.

    A Sell signal is generated when the 14-period RSI crosses below a bearish threshold (40),
    signaling strengthening downward momentum.

    Args:
        data (MarketData): An object containing market data for various currency pairs.

    Returns:
        list[BuySignal | SellSignal]: A list of generated Buy or Sell signals.
    """
    signals: list[BuySignal | SellSignal] = []
    rule_id = "8d5c6a19-26d1-4b6d-b83a-a15ae32e2e7d"

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure candles are sorted chronologically by their hour.
        # The `warm` list is typically maintained in sorted order, but explicit sorting
        # adds robustness against potential data ordering inconsistencies.
        sorted_candles = sorted(warm_candles, key=lambda c: c.hour)

        # Check for sufficient data to calculate RSI and perform a crossover comparison.
        if len(sorted_candles) < MIN_CANDLES_FOR_RSI:
            continue

        # Extract closing prices from the sorted candles for RSI calculation.
        prices = np.array([c.close for c in sorted_candles])

        # Calculate RSI values for the extracted prices.
        rsi_values = _calculate_rsi(prices, RSI_PERIOD)

        # Filter out NaN values from the beginning of the RSI array.
        # These NaNs exist because RSI cannot be calculated for the first `RSI_PERIOD` data points.
        valid_rsi = rsi_values[~np.isnan(rsi_values)]

        # We need at least two valid RSI values (current and previous) to check for a crossover.
        if len(valid_rsi) < 2:
            continue

        current_rsi = valid_rsi[-1]
        previous_rsi = valid_rsi[-2]

        # The `current_candle` corresponds to the last candle in `sorted_candles`,
        # which is the candle for which `current_rsi` was calculated.
        current_candle = sorted_candles[-1]

        # --- Generate Signals Based on Crossover Conditions ---

        # Buy signal condition:
        # The previous RSI was below the bullish threshold AND
        # the current RSI is at or above the bullish threshold.
        if previous_rsi < RSI_BUY_THRESHOLD and current_rsi >= RSI_BUY_THRESHOLD:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id=rule_id
            ))
        # Sell signal condition:
        # The previous RSI was above the bearish threshold AND
        # the current RSI is at or below the bearish threshold.
        elif previous_rsi > RSI_SELL_THRESHOLD and current_rsi <= RSI_SELL_THRESHOLD:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id=rule_id
            ))

    return signals