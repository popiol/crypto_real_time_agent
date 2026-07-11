from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Define parameters
SHORT_EMA_PERIOD = 10
MID_EMA_PERIOD = 20
LONG_EMA_PERIOD = 50
ADX_PERIOD = 14
ADX_THRESHOLD = 25

# Minimum number of candles required for all indicators to have at least one valid value.
# EMA requires at least its period number of candles for the first valid value.
# ADX calculation involves several smoothing steps, each requiring 'period' data points.
# Specifically, for ADX(N), we need N candles for ATR/DM smoothing, then N candles for DX smoothing,
# meaning the first valid ADX value appears after roughly 2*N - 1 candles.
# So, MIN_CANDLES_REQUIRED = max(LONG_EMA_PERIOD, 2 * ADX_PERIOD - 1).
MIN_CANDLES_REQUIRED = max(LONG_EMA_PERIOD, 2 * ADX_PERIOD - 1)


def _calculate_ema_np(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates Exponential Moving Average (EMA) using numpy.
    Returns an array of the same length as prices, with NaN for initial values
    where EMA cannot be calculated.
    """
    if len(prices) < period:
        return np.full_like(prices, np.nan, dtype=float)

    alpha = 2 / (period + 1)
    ema = np.full_like(prices, np.nan, dtype=float)

    # Initialize the first EMA value with the Simple Moving Average (SMA) of the first 'period' values
    ema[period - 1] = np.mean(prices[:period])

    # Calculate subsequent EMAs
    for i in range(period, len(prices)):
        ema[i] = prices[i] * alpha + ema[i-1] * (1 - alpha)

    return ema


def _wilders_smoothing(data: np.ndarray, period: int) -> np.ndarray:
    """
    Applies Wilder's smoothing method to a data series.
    Returns an array of the same length as data, with NaN for initial values.
    """
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)

    smoothed = np.full_like(data, np.nan, dtype=float)
    # The first smoothed value is the simple average of the first 'period' values
    smoothed[period-1] = np.sum(data[:period]) / period
    for i in range(period, len(data)):
        smoothed[i] = (smoothed[i-1] * (period - 1) + data[i]) / period
    return smoothed


def _calculate_adx_np(high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates the Average Directional Index (ADX) using numpy.
    Returns an array of the same length as close_prices, with NaN for initial values
    where ADX cannot be calculated.
    """
    n = len(close_prices)
    if n < period + 1: # Need at least period+1 candles for TR/DM calculation to start correctly
        return np.full(n, np.nan, dtype=float)

    # True Range (TR)
    tr = np.zeros(n)
    tr[0] = high_prices[0] - low_prices[0] # For the very first candle, no previous close
    for i in range(1, n):
        tr[i] = max(high_prices[i] - low_prices[i],
                    abs(high_prices[i] - close_prices[i-1]),
                    abs(low_prices[i] - close_prices[i-1]))

    # Directional Movement (+DM, -DM)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high_prices[i] - high_prices[i-1]
        down_move = low_prices[i-1] - low_prices[i]

        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0

    # Wilder's smoothing for TR, +DM, -DM
    atr = _wilders_smoothing(tr, period)
    plus_dm_smooth = _wilders_smoothing(plus_dm, period)
    minus_dm_smooth = _wilders_smoothing(minus_dm, period)

    # Directional Indicators (+DI, -DI)
    plus_di = np.full(n, np.nan, dtype=float)
    minus_di = np.full(n, np.nan, dtype=float)

    # DI values are valid from where ATR, plus_dm_smooth, minus_dm_smooth become valid
    for i in range(n):
        if not np.isnan(atr[i]) and atr[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
        # If ATR is NaN or 0, DI values remain NaN

    # Directional Index (DX)
    dx = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx[i] = (abs(plus_di[i] - minus_di[i]) / di_sum) * 100
        # If DI values are NaN or sum is 0, DX remains NaN

    # ADX (Average Directional Index) - Smoothed DX
    adx = _wilders_smoothing(dx, period)

    return adx


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trend-following signals based on Triple Moving Average Crossover
    with ADX Trend Strength Confirmation.

    A Buy signal is issued when a short-term EMA crosses above a mid-term EMA,
    which in turn is above a long-term EMA, and ADX indicates a strong trend.
    A Sell signal is triggered symmetrically.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough data for all indicator calculations
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract required price arrays from warm candles
        close_prices = np.array([c.close for c in warm_candles])
        high_prices = np.array([c.high for c in warm_candles])
        low_prices = np.array([c.low for c in warm_candles])

        # Calculate EMAs
        ema_short = _calculate_ema_np(close_prices, SHORT_EMA_PERIOD)
        ema_mid = _calculate_ema_np(close_prices, MID_EMA_PERIOD)
        ema_long = _calculate_ema_np(close_prices, LONG_EMA_PERIOD)

        # Calculate ADX
        adx_values = _calculate_adx_np(high_prices, low_prices, close_prices, ADX_PERIOD)

        # Get the latest valid indicator values for decision making
        # Check if the last element is NaN before accessing its value
        last_ema_short = ema_short[-1] if not np.isnan(ema_short[-1]) else None
        last_ema_mid = ema_mid[-1] if not np.isnan(ema_mid[-1]) else None
        last_ema_long = ema_long[-1] if not np.isnan(ema_long[-1]) else None
        last_adx = adx_values[-1] if not np.isnan(adx_values[-1]) else None

        # All indicators must have valid (non-NaN) values for a signal to be generated
        if (last_ema_short is None or last_ema_mid is None or
            last_ema_long is None or last_adx is None):
            continue

        # Get the timestamp and price from the latest candle
        latest_candle = warm_candles[-1]
        signal_timestamp = latest_candle.hour
        signal_price = latest_candle.close

        # Generate signals based on the rule logic
        # Buy Signal: EMAs stacked upwards and ADX confirms strong trend
        if (last_ema_short > last_ema_mid and
            last_ema_mid > last_ema_long and
            last_adx > ADX_THRESHOLD):
            signals.append(BuySignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
            ))
        # Sell Signal: EMAs stacked downwards and ADX confirms strong trend
        elif (last_ema_short < last_ema_mid and
              last_ema_mid < last_ema_long and
              last_adx > ADX_THRESHOLD):
            signals.append(SellSignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
            ))

    return signals