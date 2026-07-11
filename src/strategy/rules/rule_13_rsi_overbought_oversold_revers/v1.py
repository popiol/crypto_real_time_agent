from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Constants for the RSI rule
RSI_PERIOD = 14
RSI_OVERBOUGHT_THRESHOLD = 70
RSI_OVERSOLD_THRESHOLD = 30
# We need `RSI_PERIOD` candles for the initial average gain/loss calculation (SMA),
# then one more candle to calculate the first smoothed RSI value (EMA-like),
# and an additional candle to get a *previous* RSI value to detect a cross.
# So, `RSI_PERIOD + 2` candles are needed for cross detection.
MIN_CANDLES_FOR_RSI_CROSS = RSI_PERIOD + 2

def _calculate_rsi_series(prices: list[float], period: int = RSI_PERIOD) -> np.ndarray:
    """
    Calculates the Relative Strength Index (RSI) for a series of prices.
    Returns a numpy array of RSI values, starting from the point where enough
    data is available for the first calculation.
    """
    if len(prices) < period + 1:
        # Not enough data to calculate even the first RSI value
        return np.array([])

    prices_arr = np.array(prices, dtype=np.float64)
    
    # Calculate price changes (deltas)
    # deltas will have len(prices) - 1 elements
    deltas = np.diff(prices_arr)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros_like(gains)
    avg_losses = np.zeros_like(losses)

    # Calculate initial average gain and loss (SMA for the first 'period' values)
    # These correspond to the `period`-th delta, which is at index `period - 1`
    avg_gains[period - 1] = np.mean(gains[:period])
    avg_losses[period - 1] = np.mean(losses[:period])

    # Calculate subsequent average gain and loss (Wilder's smoothing method, EMA-like)
    for i in range(period, len(gains)):
        avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i]) / period
        avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i]) / period

    # Calculate Relative Strength (RS)
    rs = np.zeros_like(avg_gains)
    
    # Where avg_losses is not zero, calculate normally
    non_zero_losses = avg_losses != 0
    rs[non_zero_losses] = avg_gains[non_zero_losses] / avg_losses[non_zero_losses]
    
    # Handle cases where avg_losses is zero:
    zero_losses_mask = ~non_zero_losses # Where avg_losses == 0

    # If avg_losses is zero AND avg_gains is also zero, RS is typically 1 (RSI=50, no change)
    rs[zero_losses_mask & (avg_gains == 0)] = 1 
    
    # If avg_losses is zero BUT avg_gains is non-zero, RS is infinite (RSI=100, only gains)
    rs[zero_losses_mask & (avg_gains != 0)] = np.inf

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Explicitly set RSI to 100 where RS was infinite (only gains)
    rsi[rs == np.inf] = 100

    # Return only the valid RSI values, which start from the `period - 1` index of `avg_gains`/`avg_losses`
    # as these are the first points where a full `period` window is available for smoothing.
    return rsi[period - 1:]


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the RSI Overbought/Oversold Reversion trading rule.

    A Buy signal is generated when the RSI falls below a specified oversold threshold
    from above. A Sell signal is generated when the RSI rises above a specified
    overbought threshold from below.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure we have enough candles to calculate RSI for cross detection
        if len(warm_candles) < MIN_CANDLES_FOR_RSI_CROSS:
            continue

        # Extract closing prices from the warm candles
        close_prices = [candle.close for candle in warm_candles]

        # Calculate the RSI series
        rsi_series = _calculate_rsi_series(close_prices, RSI_PERIOD)

        # We need at least two RSI values to detect a cross
        if len(rsi_series) < 2:
            continue

        current_rsi = rsi_series[-1]
        previous_rsi = rsi_series[-2]
        latest_candle = warm_candles[-1] # This candle corresponds to current_rsi

        # Buy signal: RSI crosses below oversold threshold from above
        if current_rsi < RSI_OVERSOLD_THRESHOLD and previous_rsi >= RSI_OVERSOLD_THRESHOLD:
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_candle.hour,
                price=latest_candle.close,
                rule_id="RSI_Reversion",
                confidence=None # No specific confidence defined in the rule
            ))
        # Sell signal: RSI crosses above overbought threshold from below
        elif current_rsi > RSI_OVERBOUGHT_THRESHOLD and previous_rsi <= RSI_OVERBOUGHT_THRESHOLD:
            signals.append(SellSignal(
                pair=pair,
                timestamp=latest_candle.hour,
                price=latest_candle.close,
                rule_id="RSI_Reversion",
                confidence=None # No specific confidence defined in the rule
            ))

    return signals