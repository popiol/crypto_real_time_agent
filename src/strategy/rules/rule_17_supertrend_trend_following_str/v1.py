from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Parameters for Supertrend indicator
ATR_PERIOD = 10
MULTIPLIER = 3.0

# Minimum number of candles required to calculate Supertrend
# We need ATR_PERIOD candles for the initial ATR calculation,
# and then at least one more candle for the Supertrend line itself.
MIN_CANDLES_REQUIRED = ATR_PERIOD + 1


def _calculate_supertrend(
    high: list[float], low: list[float], close: list[float], atr_period: int, multiplier: float
) -> tuple[float, str | None]:
    """
    Calculates the Supertrend indicator values for the given price data.
    Returns the latest Supertrend line value and its corresponding trend state.

    Args:
        high: A list of high prices.
        low: A list of low prices.
        close: A list of close prices.
        atr_period: The period for calculating Average True Range (ATR).
        multiplier: The multiplier for ATR to determine band width.

    Returns:
        A tuple containing:
        - The latest Supertrend line value (float).
        - The latest trend state ('UPTREND', 'DOWNTREND', or None).
        Returns (np.nan, None) if insufficient data.
    """
    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)
    close = np.array(close, dtype=float)
    n = len(close)

    if n < atr_period + 1:
        return np.nan, None

    # 1. Calculate True Range (TR)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]  # First TR is just high-low
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # 2. Calculate Average True Range (ATR)
    atr = np.zeros(n)
    # The first ATR value (at index atr_period-1) is a simple moving average of TR
    atr[atr_period - 1] = np.mean(tr[:atr_period])
    # Subsequent ATR values use an exponential moving average formula
    for i in range(atr_period, n):
        atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

    # 3. Calculate Basic Upper and Lower Bands
    basic_upper_band = (high + low) / 2 + multiplier * atr
    basic_lower_band = (high + low) / 2 - multiplier * atr

    # 4. Initialize Supertrend line and trend state arrays
    supertrend_line = np.zeros(n)
    current_trend_state = [None] * n  # Stores 'UPTREND' or 'DOWNTREND'

    # Initialize for the first valid ATR point (index atr_period-1)
    # Determine initial trend based on close price relative to bands
    if close[atr_period - 1] > basic_upper_band[atr_period - 1]:
        current_trend_state[atr_period - 1] = 'UPTREND'
        supertrend_line[atr_period - 1] = basic_lower_band[atr_period - 1]
    elif close[atr_period - 1] < basic_lower_band[atr_period - 1]:
        current_trend_state[atr_period - 1] = 'DOWNTREND'
        supertrend_line[atr_period - 1] = basic_upper_band[atr_period - 1]
    else:
        # If price is within bands, default to uptrend and lower band for initialization
        current_trend_state[atr_period - 1] = 'UPTREND'
        supertrend_line[atr_period - 1] = basic_lower_band[atr_period - 1]

    # 5. Calculate Supertrend line and trend state for subsequent points
    for i in range(atr_period, n):
        prev_trend_state = current_trend_state[i - 1]
        prev_supertrend_line = supertrend_line[i - 1]

        if prev_trend_state == 'UPTREND':
            # If current close drops below the previous Supertrend line, trend changes to DOWNTREND
            if close[i] < prev_supertrend_line:
                current_trend_state[i] = 'DOWNTREND'
                supertrend_line[i] = basic_upper_band[i]  # New Supertrend line is the current upper band
            else:
                # Still in uptrend, Supertrend line is the maximum of current lower band and previous Supertrend line
                current_trend_state[i] = 'UPTREND'
                supertrend_line[i] = max(basic_lower_band[i], prev_supertrend_line)
        else:  # prev_trend_state == 'DOWNTREND'
            # If current close rises above the previous Supertrend line, trend changes to UPTREND
            if close[i] > prev_supertrend_line:
                current_trend_state[i] = 'UPTREND'
                supertrend_line[i] = basic_lower_band[i]  # New Supertrend line is the current lower band
            else:
                # Still in downtrend, Supertrend line is the minimum of current upper band and previous Supertrend line
                current_trend_state[i] = 'DOWNTREND'
                supertrend_line[i] = min(basic_upper_band[i], prev_supertrend_line)

    # Return the latest Supertrend line and trend state
    return supertrend_line[-1], current_trend_state[-1]


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates Buy/Sell signals based on the Supertrend indicator.

    A Buy signal is generated when the price closes above the Supertrend line,
    indicating an uptrend. A Sell signal is generated when the price closes
    below the Supertrend line, indicating a downtrend.

    Args:
        data: A MarketData object containing warm candles for various currency pairs.

    Returns:
        A list of BuySignal or SellSignal objects.
    """
    signals: list[BuySignal | SellSignal] = []
    rule_id = "b7ef9b37-6278-4759-a2ff-0830375d2904"

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract high, low, close prices from warm candles
        high_prices = [c.high for c in warm_candles]
        low_prices = [c.low for c in warm_candles]
        close_prices = [c.close for c in warm_candles]

        # Calculate Supertrend for the current pair
        supertrend_line, current_trend_state = _calculate_supertrend(
            high_prices, low_prices, close_prices, ATR_PERIOD, MULTIPLIER
        )

        # Skip if Supertrend calculation failed (e.g., due to insufficient data)
        if np.isnan(supertrend_line) or current_trend_state is None:
            continue

        # Get the latest close price and timestamp
        current_close = close_prices[-1]
        timestamp = warm_candles[-1].hour

        # Signal Generation Logic
        if current_trend_state == 'UPTREND' and current_close > supertrend_line:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=timestamp,
                    price=current_close,
                    rule_id=rule_id,
                    confidence=1.0  # Assign a default confidence
                )
            )
        elif current_trend_state == 'DOWNTREND' and current_close < supertrend_line:
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=timestamp,
                    price=current_close,
                    rule_id=rule_id,
                    confidence=1.0  # Assign a default confidence
                )
            )

    return signals