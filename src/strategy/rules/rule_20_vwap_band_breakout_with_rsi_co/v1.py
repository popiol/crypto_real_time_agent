from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle
from datetime import datetime

# Rule ID for identification
RULE_ID = "c5686b23-a52f-479a-9ac3-39f7c964335e"

# Parameters
PERIOD_VWAP = 20
STD_DEV_MULTIPLIER = 2
PERIOD_RSI = 14
RSI_BUY_LOWER = 50
RSI_BUY_UPPER = 70
RSI_SELL_LOWER = 30
RSI_SELL_UPPER = 50

# Minimum number of candles required for calculations
# RSI needs PERIOD_RSI + 1 candles for its initial calculation (first 'change' requires 2 points)
# VWAP (SMA) needs PERIOD_VWAP candles.
MIN_CANDLES_REQUIRED = max(PERIOD_VWAP, PERIOD_RSI + 1)

def _calculate_rsi(prices: np.ndarray, period: int) -> float:
    """Calculates the Relative Strength Index (RSI) for a given price series."""
    if len(prices) <= period:
        return np.nan

    # Calculate price changes
    changes = prices[1:] - prices[:-1]

    # Separate gains and losses
    gains = np.maximum(0, changes)
    losses = np.abs(np.minimum(0, changes))

    # Calculate initial average gain and loss over the period
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Apply Wilder's smoothing for subsequent values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Calculate Relative Strength (RS)
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0  # Handle division by zero
    
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calculate_vwap_bands_approx(closes: np.ndarray, period: int, std_multiplier: float) -> tuple[float, float, float]:
    """
    Calculates VWAP (approximated as SMA) and its standard deviation bands.
    
    Note: The original pseudocode for VWAP (Volume-Weighted Average Price) requires 'Volume' data.
    As the `WarmCandle` model does not provide volume, this implementation uses a Simple Moving Average (SMA)
    of 'Close' prices as an approximation for VWAP, and calculates standard deviation bands around this SMA.
    This maintains the structural intent of the rule (price breaking bands with RSI confirmation)
    but without the volume-weighting aspect of true VWAP.
    """
    if len(closes) < period:
        return np.nan, np.nan, np.nan

    # Use the last 'period' close prices for calculations
    relevant_closes = closes[-period:]

    # VWAP approximation (Simple Moving Average)
    vwap_approx = np.mean(relevant_closes)

    # Standard deviation of closes (equivalent to std dev of closes from their mean)
    std_dev_approx = np.std(relevant_closes)

    # Calculate VWAP Bands
    upper_band = vwap_approx + (std_multiplier * std_dev_approx)
    lower_band = vwap_approx - (std_multiplier * std_dev_approx)

    return vwap_approx, upper_band, lower_band

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough warm candles are available for calculations
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Sort candles by hour to ensure chronological order (PairData.warm should be ordered, but defensive)
        warm_candles.sort(key=lambda c: c.hour)

        # Extract close prices as a numpy array for efficient calculations
        closes = np.array([c.close for c in warm_candles])

        # Get the latest candle's data
        latest_candle = warm_candles[-1]
        current_close = closes[-1]
        timestamp = latest_candle.hour

        # Calculate VWAP Bands (SMA approximation)
        vwap_approx, upper_vwap_band, lower_vwap_band = _calculate_vwap_bands_approx(
            closes, PERIOD_VWAP, STD_DEV_MULTIPLIER
        )

        # Calculate RSI
        rsi = _calculate_rsi(closes, PERIOD_RSI)

        # Skip if any calculation resulted in NaN (e.g., due to insufficient data within the helper)
        if np.isnan(vwap_approx) or np.isnan(rsi):
            continue

        # Buy Signal Condition:
        # Price breaks above upper VWAP band AND RSI is in a bullish, but not overbought, range
        if (current_close > upper_vwap_band and
                RSI_BUY_LOWER < rsi < RSI_BUY_UPPER):
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_close,
                rule_id=RULE_ID
            ))

        # Sell Signal Condition:
        # Price breaks below lower VWAP band AND RSI is in a bearish, but not oversold, range
        if (current_close < lower_vwap_band and
                RSI_SELL_LOWER < rsi < RSI_SELL_UPPER):
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_close,
                rule_id=RULE_ID
            ))

    return signals