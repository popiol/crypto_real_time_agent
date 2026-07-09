from __future__ import annotations
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle
import numpy as np

# Rule parameters
RSI_PERIOD = 14
BUY_THRESHOLD = 55
SELL_THRESHOLD = 45

# Minimum number of candles required to calculate the first RSI value.
# RSI_PERIOD + 1 candles are needed for the first RSI value (e.g., 14 changes, 15 candles).
MIN_CANDLES_FOR_RSI = RSI_PERIOD + 1

# Minimum number of candles required to detect a crossover.
# This means we need at least two RSI values, so MIN_CANDLES_FOR_RSI + 1.
MIN_CANDLES_FOR_SIGNAL = MIN_CANDLES_FOR_RSI + 1

def _calculate_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates the Relative Strength Index (RSI) for a given series of close prices.

    Args:
        closes (np.ndarray): A numpy array of closing prices.
        period (int): The RSI period (e.g., 14).

    Returns:
        np.ndarray: A numpy array of RSI values. Returns an empty array if
                    insufficient data is provided.
    """
    if len(closes) < period + 1:
        return np.array([])

    # Calculate price differences (deltas)
    diffs = np.diff(closes)

    # Separate gains (positive changes) and losses (absolute value of negative changes)
    gains = np.maximum(0, diffs)
    losses = np.maximum(0, -diffs)

    # Initialize arrays for RSI values
    rsi_values = np.zeros(len(closes) - period)

    # Calculate initial average gain and loss over the first 'period' differences
    # These differences correspond to candles from index 1 to 'period'
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Calculate the first RS and RSI value
    if avg_loss == 0:
        rs = np.inf if avg_gain > 0 else 0.0 # Handle division by zero
    else:
        rs = avg_gain / avg_loss
    rsi_values[0] = 100 - (100 / (1 + rs))

    # Calculate subsequent RSI values using smoothed averages (Wilder's smoothing)
    for i in range(period, len(diffs)):
        current_gain = gains[i]
        current_loss = losses[i]

        avg_gain = ((avg_gain * (period - 1)) + current_gain) / period
        avg_loss = ((avg_loss * (period - 1)) + current_loss) / period

        if avg_loss == 0:
            rs = np.inf if avg_gain > 0 else 0.0 # Handle division by zero
        else:
            rs = avg_gain / avg_loss
        rsi_values[i - period + 1] = 100 - (100 / (1 + rs))

    return rsi_values

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on RSI mid-range crossovers.

    A Buy signal is emitted when RSI crosses above the BUY_THRESHOLD (e.g., 55).
    A Sell signal is emitted when RSI crosses below the SELL_THRESHOLD (e.g., 45).

    Args:
        data (MarketData): A dictionary containing market data for various pairs.

    Returns:
        list[BuySignal | SellSignal]: A list of generated buy and sell signals.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough warm candles are available to calculate at least two RSI values
        if len(warm_candles) < MIN_CANDLES_FOR_SIGNAL:
            continue

        # Extract close prices and corresponding timestamps from warm candles
        closes = np.array([candle.close for candle in warm_candles])
        timestamps = [candle.hour for candle in warm_candles]

        # Calculate RSI values
        rsi_values = _calculate_rsi(closes, RSI_PERIOD)

        # If RSI calculation resulted in insufficient values for crossover detection, skip
        if len(rsi_values) < 2:
            continue

        # Iterate through the calculated RSI values to detect crossovers.
        # rsi_values[k] corresponds to warm_candles[RSI_PERIOD + k].
        # We start from the second RSI value (index 1) to compare with the previous one.
        for i in range(1, len(rsi_values)):
            current_rsi = rsi_values[i]
            previous_rsi = rsi_values[i-1]

            # The current candle being evaluated is at an index offset by RSI_PERIOD
            # because the first RSI value is for warm_candles[RSI_PERIOD].
            current_candle_index = RSI_PERIOD + i
            current_candle = warm_candles[current_candle_index]

            # Buy signal: RSI crosses above BUY_THRESHOLD
            if previous_rsi < BUY_THRESHOLD and current_rsi >= BUY_THRESHOLD:
                signals.append(BuySignal(
                    pair=pair,
                    timestamp=current_candle.hour,
                    price=current_candle.close,
                    rule_id="f047e8f2-adf5-418d-852f-0b2b3a611571",
                ))
            # Sell signal: RSI crosses below SELL_THRESHOLD
            elif previous_rsi > SELL_THRESHOLD and current_rsi <= SELL_THRESHOLD:
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=current_candle.hour,
                    price=current_candle.close,
                    rule_id="f047e8f2-adf5-418d-852f-0b2b3a611571",
                ))

    return signals