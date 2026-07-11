from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Rule parameters
VWMA_SHORT_PERIOD = 10
VWMA_LONG_PERIOD = 30
OBV_SMA_PERIOD = 9

# Minimum number of warm candles required to calculate all indicators
# We need enough candles for the longest VWMA period (VWMA_LONG_PERIOD)
# and for the OBV_SMA period (OBV_SMA_PERIOD).
# To check for crossovers and trends, we need at least the current and previous indicator values.
# For an indicator of period P, we need P values for the current period, and P+1 for the previous period.
# OBV itself starts calculation from the second candle. OBV_SMA then needs OBV_SMA_PERIOD values of OBV.
# So, for OBV_SMA[-2] to be valid, we need OBV_SMA_PERIOD + 1 values of OBV, which means OBV_SMA_PERIOD + 1 candles.
# Thus, the minimum required candles is the maximum of (longest VWMA period + 1) and (OBV_SMA period + 1).
MIN_CANDLES_REQUIRED = max(VWMA_LONG_PERIOD + 1, OBV_SMA_PERIOD + 1)

def _calculate_vwma(prices: np.ndarray, volumes: np.ndarray, period: int) -> np.ndarray:
    """Calculates Volume-Weighted Moving Average."""
    if len(prices) < period:
        return np.full(len(prices), np.nan)

    vwmas = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        window_prices = prices[i - period + 1 : i + 1]
        window_volumes = volumes[i - period + 1 : i + 1]
        sum_volumes = np.sum(window_volumes)
        if sum_volumes > 0:
            vwmas[i] = np.sum(window_prices * window_volumes) / sum_volumes
        else:
            vwmas[i] = np.nan # Avoid division by zero if all volumes are 0
    return vwmas

def _calculate_obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Calculates On-Balance Volume.
    OBV[i] = OBV[i-1] + Volume[i] if Close[i] > Close[i-1]
    OBV[i] = OBV[i-1] - Volume[i] if Close[i] < Close[i-1]
    OBV[i] = OBV[i-1] if Close[i] == Close[i-1]
    Initial OBV is typically 0.
    """
    obv = np.zeros(len(closes), dtype=np.float64)
    if len(closes) > 1:
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
    return obv

def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < period:
        return np.full(len(data), np.nan)

    # Using convolution for efficiency
    weights = np.ones(period) / period
    sma = np.convolve(data, weights, mode='valid')
    # Pad with NaN at the beginning to align with original data length
    return np.concatenate((np.full(period - 1, np.nan), sma))

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract closing prices and volumes into numpy arrays for efficient calculations
        closes = np.array([c.close for c in warm_candles], dtype=np.float64)
        volumes = np.array([c.volume for c in warm_candles], dtype=np.float64)

        if len(closes) == 0 or len(volumes) == 0:
            continue

        # Calculate indicators
        vwma_short = _calculate_vwma(closes, volumes, VWMA_SHORT_PERIOD)
        vwma_long = _calculate_vwma(closes, volumes, VWMA_LONG_PERIOD)
        obv = _calculate_obv(closes, volumes)
        obv_sma = _calculate_sma(obv, OBV_SMA_PERIOD)

        # Ensure all necessary indicator values for the current and previous period are valid (not NaN)
        # We need vwma_short[-1], vwma_short[-2], vwma_long[-1], vwma_long[-2], obv[-1], obv_sma[-1]
        # And obv_sma[-2] for the OBV trend confirmation.
        if (np.isnan(vwma_short[-1]) or np.isnan(vwma_short[-2]) or
            np.isnan(vwma_long[-1]) or np.isnan(vwma_long[-2]) or
            np.isnan(obv_sma[-1]) or np.isnan(obv_sma[-2])):
            continue

        # Get current and previous values for comparison
        current_vwma_short = vwma_short[-1]
        prev_vwma_short = vwma_short[-2]
        current_vwma_long = vwma_long[-1]
        prev_vwma_long = vwma_long[-2]
        current_obv = obv[-1]
        current_obv_sma = obv_sma[-1]

        # Get the timestamp and price of the latest candle for the signal
        latest_candle = warm_candles[-1]
        timestamp = latest_candle.hour
        price = latest_candle.close

        # Buy Signal Condition:
        # 1. Short VWMA crosses above Long VWMA
        # 2. OBV is currently above its SMA (upward trend confirmation)
        vwma_short_cross_up = (current_vwma_short > current_vwma_long) and (prev_vwma_short <= prev_vwma_long)
        obv_trend_up = (current_obv > current_obv_sma)

        if vwma_short_cross_up and obv_trend_up:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=price,
                rule_id="53fc2ea0-c55f-459c-93d2-923ba5e47c23",
                confidence=0.81
            ))

        # Sell Signal Condition:
        # 1. Short VWMA crosses below Long VWMA
        # 2. OBV is currently below its SMA (downward trend confirmation)
        vwma_short_cross_down = (current_vwma_short < current_vwma_long) and (prev_vwma_short >= prev_vwma_long)
        obv_trend_down = (current_obv < current_obv_sma)

        if vwma_short_cross_down and obv_trend_down:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=price,
                rule_id="53fc2ea0-c55f-459c-93d2-923ba5e47c23",
                confidence=0.81
            ))

    return signals