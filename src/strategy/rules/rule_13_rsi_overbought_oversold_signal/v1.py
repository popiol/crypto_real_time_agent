from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

# --- Rule Constants ---
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
# To calculate 14-period RSI, we need at least 15 candles
# (14 price changes to establish initial average gain/loss for the first RSI value).
MIN_CANDLES_FOR_RSI = RSI_PERIOD + 1

# --- Rule ID ---
RULE_ID = "b78b07f1-2a99-4e60-8ee4-6a3dcd1c6e47"

def _calculate_rsi(prices: list[float], period: int) -> float | None:
    """
    Calculates the Relative Strength Index (RSI) for a given list of prices and period.
    Uses Wilder's smoothing method for subsequent average gain/loss calculations.
    Returns the RSI value for the most recent price, or None if insufficient data.
    """
    if len(prices) < period + 1:
        return None

    np_prices = np.array(prices, dtype=float)
    # Calculate price differences (P_i - P_{i-1})
    price_diffs = np_prices[1:] - np_prices[:-1]

    # Calculate gains (positive differences) and losses (absolute value of negative differences)
    gains = np.where(price_diffs > 0, price_diffs, 0)
    losses = np.where(price_diffs < 0, -price_diffs, 0)

    # Initial average gain and loss using SMA for the first 'period' differences
    # These averages correspond to the RSI for the 'period'-th candle (index `period` in original prices)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Calculate subsequent average gain/loss using Wilder's smoothing
    # The loop starts from the (period+1)-th difference (index `period` in `gains`/`losses` arrays)
    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    # Calculate Relative Strength (RS)
    if avg_loss == 0:
        rs = np.inf if avg_gain > 0 else 1.0  # Handle cases with no losses (RSI=100) or no movement (RSI=50)
    else:
        rs = avg_gain / avg_loss

    # Calculate RSI
    if rs == np.inf:
        rsi = 100.0
    else:
        rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates Buy or Sell signals based on the 14-period Relative Strength Index (RSI).
    A Buy signal is generated when RSI falls below 30 (oversold).
    A Sell signal is generated when RSI rises above 70 (overbought).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Use warm (hourly) candles for RSI calculation
        warm_candles = pair_data.warm

        # Ensure enough data is available to calculate RSI
        if len(warm_candles) < MIN_CANDLES_FOR_RSI:
            continue

        # Extract closing prices from the candles
        prices = [candle.close for candle in warm_candles]

        # Calculate the RSI for the most recent candle
        current_rsi = _calculate_rsi(prices, RSI_PERIOD)

        if current_rsi is None:
            continue  # Should not happen if len(warm_candles) check passes, but good for robustness

        # Get the latest candle's information for signal generation
        latest_candle = warm_candles[-1]
        signal_timestamp = latest_candle.hour
        signal_price = latest_candle.close

        # Generate signals based on RSI thresholds
        if current_rsi < RSI_OVERSOLD:
            signals.append(BuySignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
                rule_id=RULE_ID,
                confidence=None  # No confidence specified in the rule
            ))
        elif current_rsi > RSI_OVERBOUGHT:
            signals.append(SellSignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
                rule_id=RULE_ID,
                confidence=None  # No confidence specified in the rule
            ))
    return signals