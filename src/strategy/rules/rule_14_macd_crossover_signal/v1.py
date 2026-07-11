from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Constants for MACD calculation periods
FAST_EMA_PERIOD = 12
SLOW_EMA_PERIOD = 26
SIGNAL_LINE_PERIOD = 9

# Minimum number of warm candles required to calculate all indicators and a crossover.
# To get the first valid Signal Line value, we need:
# SLOW_EMA_PERIOD candles for the initial 26-period EMA.
# Then, SIGNAL_LINE_PERIOD values of MACD to calculate the Signal Line.
# So, the index of the first valid Signal Line value is (SLOW_EMA_PERIOD - 1) + (SIGNAL_LINE_PERIOD - 1).
# We need at least two such valid points for a crossover check (current and previous).
# Therefore, MIN_CANDLES = (SLOW_EMA_PERIOD - 1) + (SIGNAL_LINE_PERIOD - 1) + 2 = SLOW_EMA_PERIOD + SIGNAL_LINE_PERIOD.
MIN_CANDLES = SLOW_EMA_PERIOD + SIGNAL_LINE_PERIOD # 26 + 9 = 35

def _calculate_ema_series(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates the Exponential Moving Average (EMA) series for a given period.
    The first EMA value is initialized with the Simple Moving Average (SMA)
    of the first `period` prices. Subsequent EMAs are calculated using the standard formula.
    Returns an array of EMAs, aligned such that `ema[i]` corresponds to `prices[i]`.
    Initial `period-1` values will be NaN.
    """
    if len(prices) < period:
        return np.full_like(prices, np.nan)

    alpha = 2 / (period + 1)
    ema_values = np.full_like(prices, np.nan)

    # Initialize the first EMA with the SMA of the first 'period' prices
    ema_values[period - 1] = np.mean(prices[:period])

    # Calculate subsequent EMAs
    for i in range(period, len(prices)):
        ema_values[i] = (prices[i] * alpha) + (ema_values[i-1] * (1 - alpha))
        
    return ema_values

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates Buy/Sell signals based on the MACD line crossing above or below its signal line.

    A Buy signal is triggered when the MACD line crosses above the signal line.
    A Sell signal is triggered when the MACD line crosses below the signal line.

    Args:
        data: A MarketData object containing historical price data.

    Returns:
        A list of BuySignal or SellSignal objects.
    """
    signals: list[BuySignal | SellSignal] = []
    
    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for MACD and Signal Line calculation
        warm_candles: list[WarmCandle] = pair_data.warm
        if len(warm_candles) < MIN_CANDLES:
            continue
            
        # Extract closing prices and timestamps
        close_prices = np.array([c.close for c in warm_candles], dtype=float)
        timestamps = [c.hour for c in warm_candles]

        # 1. Calculate the 12-period Exponential Moving Average (EMA) of close prices
        ema_12 = _calculate_ema_series(close_prices, FAST_EMA_PERIOD)

        # 2. Calculate the 26-period EMA of close prices
        ema_26 = _calculate_ema_series(close_prices, SLOW_EMA_PERIOD)

        # 3. Calculate the MACD line: MACD = 12-period EMA - 26-period EMA
        # The resulting MACD will have NaNs where either EMA was NaN, ensuring proper alignment.
        macd_line = ema_12 - ema_26

        # 4. Calculate the 9-period EMA of the MACD line, which is the Signal Line.
        # This will also have leading NaNs based on the MACD line's valid range.
        signal_line = _calculate_ema_series(macd_line, SIGNAL_LINE_PERIOD)
        
        # Find indices where both MACD and Signal Line are valid (not NaN)
        valid_indices = np.where(~np.isnan(macd_line) & ~np.isnan(signal_line))[0]
        
        # We need at least two valid points (current and previous) to detect a crossover
        if len(valid_indices) < 2:
            continue

        # Get the latest two valid MACD and Signal Line values and their corresponding data
        last_valid_idx = valid_indices[-1]
        second_to_last_valid_idx = valid_indices[-2]

        macd_current = macd_line[last_valid_idx]
        signal_current = signal_line[last_valid_idx]

        macd_previous = macd_line[second_to_last_valid_idx]
        signal_previous = signal_line[second_to_last_valid_idx]
        
        current_timestamp = timestamps[last_valid_idx]
        current_price = close_prices[last_valid_idx]

        # 5. Generate a Buy signal if the MACD line crosses above the Signal Line.
        # This occurs when MACD was below or equal to Signal in the previous period,
        # and is now above the Signal line.
        if macd_current > signal_current and macd_previous <= signal_previous:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id="2a2e8eaa-cac9-4112-b069-a0e67c1415b7",
                confidence=None 
            ))
        # 6. Generate a Sell signal if the MACD line crosses below the Signal Line.
        # This occurs when MACD was above or equal to Signal in the previous period,
        # and is now below the Signal line.
        elif macd_current < signal_current and macd_previous >= signal_previous:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id="2a2e8eaa-cac9-4112-b069-a0e67c1415b7",
                confidence=None 
            ))
            
    return signals