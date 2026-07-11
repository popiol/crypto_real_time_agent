from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Rule parameters
RULE_ID = "6d0bb84a-4ee8-4db3-903b-e470801e26d2"
SMA_SHORT_PERIOD = 5  # Adjusted from 10 to fit typical `warm` data length (max 24 candles)
SMA_LONG_PERIOD = 15  # Adjusted from 30 to fit typical `warm` data length (max 24 candles)
ADX_PERIOD = 10       # Adjusted from 14 to fit typical `warm` data length (needs 2*P candles for previous ADX value)
ADX_THRESHOLD = 25.0

# Minimum number of candles required for calculations to get valid (non-NaN) indicator values
# for both the current and previous candle, which is necessary for crossover detection.
# 1. For SMA: To get a valid SMA for `N` candles, `N` prices are needed. To get `sma[-2]` valid, `N+1` candles.
#    So, `SMA_LONG_PERIOD + 1` candles are needed for SMAs.
# 2. For ADX: To get the first valid ADX value (at index `2*P-2`), `2*P-1` candles are needed.
#    To get the second valid ADX value (at index `2*P-1`), `2*P` candles are needed.
#    So, `2 * ADX_PERIOD` candles are needed for ADX.
MIN_CANDLES_FOR_SIGNAL = max(SMA_LONG_PERIOD + 1, 2 * ADX_PERIOD)


def _calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(prices) < period:
        return np.full(len(prices), np.nan)  # Return NaNs for all if not enough data

    # Convolve returns an array of length len(prices) - period + 1
    sma_values = np.convolve(prices, np.ones(period) / period, mode='valid')

    # Pad with NaNs at the beginning to align with original prices
    return np.concatenate((np.full(period - 1, np.nan), sma_values))


def _calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculates Average Directional Index (ADX) using Wilder's smoothing."""
    num_candles = len(high)
    if num_candles < 2:
        return np.full(num_candles, np.nan)

    # Initialize all arrays with NaN, then fill
    tr = np.full(num_candles, np.nan)
    plus_dm = np.full(num_candles, np.nan)
    minus_dm = np.full(num_candles, np.nan)

    # Calculate True Range (TR) and Directional Movement (+DM, -DM)
    tr[0] = high[0] - low[0]  # First TR is just high-low, as no previous close
    for i in range(1, num_candles):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
            minus_dm[i] = 0
        elif down_move > up_move and down_move > 0:
            plus_dm[i] = 0
            minus_dm[i] = down_move
        else:
            plus_dm[i] = 0
            minus_dm[i] = 0

    # Wilder's smoothing function
    def wilders_smoothing(arr: np.ndarray, smoothing_period: int) -> np.ndarray:
        if len(arr) < smoothing_period:
            return np.full(len(arr), np.nan)

        smoothed_arr = np.full(len(arr), np.nan)

        # Initial sum for the first `smoothing_period` values (from index 0 to `smoothing_period-1`)
        # This sum is the first valid smoothed value, stored at index `smoothing_period-1`.
        smoothed_arr[smoothing_period - 1] = np.sum(arr[:smoothing_period])

        for i in range(smoothing_period, len(arr)):
            smoothed_arr[i] = smoothed_arr[i - 1] - (smoothed_arr[i - 1] / smoothing_period) + arr[i]

        return smoothed_arr

    smoothed_tr = wilders_smoothing(tr, period)
    smoothed_plus_dm = wilders_smoothing(plus_dm, period)
    smoothed_minus_dm = wilders_smoothing(minus_dm, period)

    # Calculate +DI, -DI
    plus_di = np.full(num_candles, np.nan)
    minus_di = np.full(num_candles, np.nan)

    # Valid smoothed values start from index `period - 1`
    for i in range(period - 1, num_candles):
        if smoothed_tr[i] != 0:
            plus_di[i] = (smoothed_plus_dm[i] / smoothed_tr[i]) * 100
            minus_di[i] = (smoothed_minus_dm[i] / smoothed_tr[i]) * 100
        else:
            plus_di[i] = 0
            minus_di[i] = 0

    # Calculate DX
    dx = np.full(num_candles, np.nan)
    # Valid DI values start from index `period - 1`
    for i in range(period - 1, num_candles):
        sum_di = plus_di[i] + minus_di[i]
        if sum_di != 0:
            dx[i] = (abs(plus_di[i] - minus_di[i]) / sum_di) * 100
        else:
            dx[i] = 0

    # Smooth DX to get ADX
    adx = np.full(num_candles, np.nan)

    # The first ADX value is the simple average of the first `period` valid DX values.
    # These DX values are available from index `period-1` up to `2*period-2`.
    # The first ADX value itself will be at index `2*period-2`.
    if num_candles >= 2 * period - 1:
        # Extract the `period` DX values needed for the initial ADX average
        # These are dx[period-1], dx[period], ..., dx[2*period-2]
        first_dx_segment = dx[period - 1: 2 * period - 1]

        # Ensure that this segment has `period` non-NaN values
        if len(first_dx_segment) == period and not np.any(np.isnan(first_dx_segment)):
            adx[2 * period - 2] = np.mean(first_dx_segment)

            # Apply Wilder's smoothing for subsequent ADX values
            for i in range(2 * period - 1, num_candles):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # Ensure enough candles are available for current and previous indicator values to be valid.
        if len(candles) < MIN_CANDLES_FOR_SIGNAL:
            continue

        # Extract required data
        close_prices = np.array([c.close for c in candles], dtype=float)
        high_prices = np.array([c.high for c in candles], dtype=float)
        low_prices = np.array([c.low for c in candles], dtype=float)
        timestamps = [c.hour for c in candles]

        # Calculate SMAs
        sma_short = _calculate_sma(close_prices, SMA_SHORT_PERIOD)
        sma_long = _calculate_sma(close_prices, SMA_LONG_PERIOD)

        # Calculate ADX
        adx_values = _calculate_adx(high_prices, low_prices, close_prices, ADX_PERIOD)

        # Check for NaN values at the critical points (last two candles).
        # This acts as a safeguard even if MIN_CANDLES_FOR_SIGNAL is correctly calculated,
        # handling cases of sparse or malformed data.
        required_values = [
            sma_short[-1], sma_short[-2],
            sma_long[-1], sma_long[-2],
            adx_values[-1], adx_values[-2]
        ]
        if any(np.isnan(required_values)):
            continue

        # Get current and previous values for signal generation
        current_sma_short = sma_short[-1]
        current_sma_long = sma_long[-1]
        current_adx = adx_values[-1]

        previous_sma_short = sma_short[-2]
        previous_sma_long = sma_long[-2]

        # Check for Buy Signal
        # SMA_short crosses above SMA_long AND ADX > ADX_threshold
        if (previous_sma_short <= previous_sma_long and
                current_sma_short > current_sma_long and
                current_adx > ADX_THRESHOLD):

            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamps[-1],
                price=close_prices[-1],
                rule_id=RULE_ID
            ))

        # Check for Sell Signal
        # SMA_short crosses below SMA_long AND ADX > ADX_threshold
        elif (previous_sma_short >= previous_sma_long and
              current_sma_short < current_sma_long and
              current_adx > ADX_THRESHOLD):

            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamps[-1],
                price=close_prices[-1],
                rule_id=RULE_ID
            ))

    return signals