from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule Parameters ---
RSI_PERIOD = 14
SMA_PERIOD = 200
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# --- Helper Functions for Indicators ---

def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(prices) < period:
        return np.array([])
    # np.convolve with 'valid' mode returns an array of length N - period + 1
    # The last element corresponds to the SMA of the last 'period' prices.
    return np.convolve(prices, np.ones(period), 'valid') / period

def calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates Relative Strength Index.
    Returns RSI values corresponding to prices[period:]
    """
    if len(prices) < period + 1:
        # Need at least 'period + 1' prices to calculate the first 'period' deltas
        # and then the first RSI value.
        return np.array([])

    deltas = np.diff(prices) # length = len(prices) - 1
    
    # Separate gains and losses
    up = np.maximum(0, deltas)
    down = np.maximum(0, -deltas)

    # Initialize average gain and loss for the first 'period' deltas
    # These averages correspond to the RSI value at prices[period]
    avg_gain = np.mean(up[:period])
    avg_loss = np.mean(down[:period])

    # The number of valid RSI values will be len(prices) - period
    rsi_values = np.zeros(len(prices) - period)

    # Calculate the first RSI value
    if avg_loss == 0:
        rs = np.inf
    else:
        rs = avg_gain / avg_loss
    rsi_values[0] = 100 - (100 / (1 + rs))

    # Iterate to calculate subsequent RSI values using Wilder's smoothing method
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + up[i]) / period
        avg_loss = (avg_loss * (period - 1) + down[i]) / period
        
        if avg_loss == 0:
            rs = np.inf
        else:
            rs = avg_gain / avg_loss
        
        # Store RSI value; rsi_values[i - period + 1] aligns with prices[i+1]
        rsi_values[i - period + 1] = 100 - (100 / (1 + rs))
        
    return rsi_values

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    
    # Minimum required candles for SMA and RSI.
    # SMA needs SMA_PERIOD candles for the last value.
    # RSI needs RSI_PERIOD + 1 candles for the first RSI, and an additional candle for PreviousRSI.
    # So, max(SMA_PERIOD, RSI_PERIOD + 2) candles are required.
    MIN_CANDLES_REQUIRED = max(SMA_PERIOD, RSI_PERIOD + 2)

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            # Insufficient data to calculate indicators as specified
            continue

        # Extract close prices from warm candles
        close_prices = np.array([c.close for c in warm_candles])
        
        # Get the latest candle's data for current price and timestamp
        last_candle: WarmCandle = warm_candles[-1]
        current_close = last_candle.close
        timestamp = last_candle.hour # Use candle close time as signal timestamp
        price_for_signal = current_close # Use candle close price for signal

        # Calculate SMA
        sma_values = calculate_sma(close_prices, SMA_PERIOD)
        if len(sma_values) == 0:
            # This should ideally not happen if MIN_CANDLES_REQUIRED is met and calculate_sma handles it.
            continue
        current_sma = sma_values[-1] # The last SMA value corresponds to the last close price

        # Calculate RSI
        rsi_values = calculate_rsi(close_prices, RSI_PERIOD)
        if len(rsi_values) < 2:
            # Need at least current and previous RSI for cross-over detection
            continue
        current_rsi = rsi_values[-1] # The last RSI value corresponds to the last close price
        previous_rsi = rsi_values[-2] # The second to last RSI value corresponds to the second to last close price

        # --- Generate Signals ---
        # The pseudocode describes a stateful system (POSITION_OPENED).
        # This stateless function generates all potential entry/exit signals.
        # The consuming trading system is responsible for managing positions and acting on these signals.

        # Long Entry Condition: RSI crosses below oversold (30) AND price is in uptrend (above SMA)
        if (current_rsi < RSI_OVERSOLD and previous_rsi >= RSI_OVERSOLD and
                current_close > current_sma):
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=price_for_signal,
                rule_id="RSI-SMA-Trend-Filter-001",
                confidence=1.0
            ))
        
        # Long Exit Condition: RSI crosses above overbought (70) OR price falls below SMA
        # This signals to close a long position.
        if ((current_rsi > RSI_OVERBOUGHT and previous_rsi <= RSI_OVERBOUGHT) or
                current_close < current_sma):
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=price_for_signal,
                rule_id="RSI-SMA-Trend-Filter-001",
                confidence=1.0
            ))

        # Short Entry Condition: RSI crosses above overbought (70) AND price is in downtrend (below SMA)
        # This signals to open a short position.
        if (current_rsi > RSI_OVERBOUGHT and previous_rsi <= RSI_OVERBOUGHT and
                current_close < current_sma):
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=price_for_signal,
                rule_id="RSI-SMA-Trend-Filter-001",
                confidence=1.0
            ))

        # Short Exit Condition: RSI crosses below oversold (30) OR price rises above SMA
        # This signals to close a short position.
        if ((current_rsi < RSI_OVERSOLD and previous_rsi >= RSI_OVERSOLD) or
                current_close > current_sma):
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=price_for_signal,
                rule_id="RSI-SMA-Trend-Filter-001",
                confidence=1.0
            ))

    return signals