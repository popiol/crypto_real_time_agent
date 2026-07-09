from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Parameters
LENGTH_MA = 20  # Length for Exponential Moving Average (EMA) for Keltner basis
LENGTH_ATR = 10 # Length for Average True Range (ATR)
MULTIPLIER = 2  # Multiplier for ATR bands
TREND_MA_LENGTH = 50 # Length for medium-term trend EMA

# Minimum number of candles required for calculations
# This needs to be sufficient for all indicators:
# - EMA for Keltner: LENGTH_MA
# - ATR: LENGTH_ATR + 1 (for True Range calculation requiring previous close)
# - EMA for Trend: TREND_MA_LENGTH
# We take the maximum of these requirements.
MIN_CANDLES = max(LENGTH_MA, TREND_MA_LENGTH, LENGTH_ATR + 1)

def _calculate_ema(prices: np.ndarray, length: int) -> np.ndarray:
    """Calculates Exponential Moving Average (EMA)."""
    if len(prices) < length:
        return np.array([])

    ema = np.zeros_like(prices, dtype=float)
    alpha = 2 / (length + 1)

    # Calculate initial SMA for the first 'length' periods
    ema[length - 1] = np.mean(prices[:length])

    # Calculate EMA for subsequent periods
    for i in range(length, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    
    # Return only the valid EMA values (from the point it's fully calculated)
    return ema[length - 1:]

def _calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, length: int) -> np.ndarray:
    """Calculates Average True Range (ATR)."""
    if len(highs) < length + 1:
        return np.array([])

    true_ranges = np.zeros(len(highs) - 1, dtype=float)
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_prev_close = abs(highs[i] - closes[i-1])
        low_prev_close = abs(lows[i] - closes[i-1])
        true_ranges[i-1] = max(high_low, high_prev_close, low_prev_close)

    # ATR is typically an EMA of True Ranges
    atr_values = _calculate_ema(true_ranges, length)
    return atr_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    rule_id = "7db5f6b5-84e9-4dc4-b18b-10f2b72e21f9"

    for pair, pair_data in data.items():
        candles = pair_data.warm

        if len(candles) < MIN_CANDLES:
            continue

        # Extract relevant data for calculations
        close_prices = np.array([c.close for c in candles])
        high_prices = np.array([c.high for c in candles])
        low_prices = np.array([c.low for c in candles])

        # Calculate Keltner Channel components
        ema_keltner_values = _calculate_ema(close_prices, LENGTH_MA)
        if len(ema_keltner_values) == 0:
            continue
        ema_keltner = ema_keltner_values[-1] # Latest Keltner EMA value

        atr_values = _calculate_atr(high_prices, low_prices, close_prices, LENGTH_ATR)
        if len(atr_values) == 0:
            continue
        current_atr = atr_values[-1] # Latest ATR value

        # Calculate Keltner Bands
        upper_band = ema_keltner + (MULTIPLIER * current_atr)
        lower_band = ema_keltner - (MULTIPLIER * current_atr)

        # Calculate Medium-term Trend EMA
        trend_ma_values = _calculate_ema(close_prices, TREND_MA_LENGTH)
        if len(trend_ma_values) == 0:
            continue
        current_trend_ma = trend_ma_values[-1] # Latest Trend MA value

        # Get current price and timestamp from the most recent candle
        current_candle = candles[-1]
        current_price = current_candle.close
        timestamp = current_candle.hour

        # Check for Buy Signal
        # Price below lower Keltner band (oversold), and overall trend is up (current price above long-term EMA)
        if current_price < lower_band and current_price > current_trend_ma:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=rule_id
            ))

        # Check for Sell Signal
        # Price above upper Keltner band (overbought), and overall trend is down (current price below long-term EMA)
        elif current_price > upper_band and current_price < current_trend_ma:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=rule_id
            ))

    return signals