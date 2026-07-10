"""Rule 43c9e05b-88ab-48c4-b86e-457b51c0db3f — ADX-Filtered Directional Trend Following."""
from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# 1. Define parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
TREND_CONFIRMATION_LOOKBACK = 3

def _calculate_adx_di(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates ADX, +DI, -DI using Wilder's smoothing method.
    The returned arrays will contain NaN values for indices where the indicator
    cannot be calculated due to insufficient preceding data.
    Specifically:
    - +DI and -DI values are valid from index `period`.
    - ADX values are valid from index `2 * period - 1`.
    """
    num_candles = len(high)
    if num_candles < period + 1: # Need at least period + 1 candles for first TR/DM
        return (
            np.full(num_candles, np.nan),
            np.full(num_candles, np.nan),
            np.full(num_candles, np.nan),
        )

    # Initialize arrays for True Range (TR) and Directional Movement (DM)
    tr_vals = np.zeros(num_candles)
    plus_dm_vals = np.zeros(num_candles)
    minus_dm_vals = np.zeros(num_candles)

    # Calculate TR, +DM, -DM for each bar starting from the second bar (index 1)
    for i in range(1, num_candles):
        # True Range
        tr_current = high[i] - low[i]
        tr_prev_close_high = abs(high[i] - close[i - 1])
        tr_prev_close_low = abs(low[i] - close[i - 1])
        tr_vals[i] = max(tr_current, tr_prev_close_high, tr_prev_close_low)

        # Directional Movement
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        plus_dm_vals[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm_vals[i] = down_move if down_move > up_move and down_move > 0 else 0

    # Initialize arrays for Smoothed TR (ATR), Smoothed +DM, Smoothed -DM
    atr_vals = np.full(num_candles, np.nan)
    plus_dms_vals = np.full(num_candles, np.nan)
    minus_dms_vals = np.full(num_candles, np.nan)

    # Calculate the first 'period' sum for smoothing
    # The sum for `period` values starts from index 1 up to `period` (inclusive).
    # So `tr_vals[1 : period + 1]` covers `period` values.
    # The first smoothed value is for the `period`-th bar (index `period`).
    atr_vals[period] = np.sum(tr_vals[1 : period + 1]) / period
    plus_dms_vals[period] = np.sum(plus_dm_vals[1 : period + 1]) / period
    minus_dms_vals[period] = np.sum(minus_dm_vals[1 : period + 1]) / period

    # Apply Wilder's smoothing for subsequent bars
    for i in range(period + 1, num_candles):
        atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr_vals[i]) / period
        plus_dms_vals[i] = (plus_dms_vals[i - 1] * (period - 1) + plus_dm_vals[i]) / period
        minus_dms_vals[i] = (minus_dms_vals[i - 1] * (period - 1) + minus_dm_vals[i]) / period

    # Initialize arrays for +DI, -DI, DX, and ADX, filled with NaN
    plus_di_arr = np.full(num_candles, np.nan)
    minus_di_arr = np.full(num_candles, np.nan)
    dx_vals = np.full(num_candles, np.nan)
    adx_arr = np.full(num_candles, np.nan)

    # Calculate +DI and -DI (valid from index `period`)
    for i in range(period, num_candles):
        if atr_vals[i] != 0:
            plus_di_arr[i] = (plus_dms_vals[i] / atr_vals[i]) * 100
            minus_di_arr[i] = (minus_dms_vals[i] / atr_vals[i]) * 100
        else:
            plus_di_arr[i] = 0.0
            minus_di_arr[i] = 0.0

    # Calculate DX (valid from index `period`)
    for i in range(period, num_candles):
        di_sum = plus_di_arr[i] + minus_di_arr[i]
        if di_sum != 0:
            dx_vals[i] = (abs(plus_di_arr[i] - minus_di_arr[i]) / di_sum) * 100
        else:
            dx_vals[i] = 0.0

    # Calculate ADX (Smoothed DX). The first ADX is the simple average of the first `period` DX values.
    # DX values are valid from `period`. So, the first `period` DX values are from index `period` to `2*period - 1`.
    # The first ADX value is therefore for the bar at index `2*period - 1`.
    if num_candles >= 2 * period:
        adx_arr[2 * period - 1] = np.sum(dx_vals[period : 2 * period]) / period
        # Subsequent smoothing for ADX
        for i in range(2 * period, num_candles):
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx_vals[i]) / period

    return adx_arr, plus_di_arr, minus_di_arr


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    # Calculate minimum required candles for valid ADX and DI lookback
    # ADX requires 2 * ADX_PERIOD candles for the first valid ADX value.
    # DI lookback requires ADX_PERIOD + TREND_CONFIRMATION_LOOKBACK candles for the earliest DI value in the window.
    # We take the maximum of these two requirements to ensure all indicators are valid for the check.
    min_candles_for_adx_valid = 2 * ADX_PERIOD
    min_candles_for_di_lookback = ADX_PERIOD + TREND_CONFIRMATION_LOOKBACK
    actual_min_candles_required = max(min_candles_for_adx_valid, min_candles_for_di_lookback)

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        if len(warm_candles) < actual_min_candles_required:
            continue

        # Extract high, low, close prices into numpy arrays
        high_prices = np.array([c.high for c in warm_candles])
        low_prices = np.array([c.low for c in warm_candles])
        close_prices = np.array([c.close for c in warm_candles])

        # Calculate ADX, +DI, -DI values
        adx_values, plus_di_values, minus_di_values = _calculate_adx_di(
            high_prices, low_prices, close_prices, ADX_PERIOD
        )

        # The current bar is the latest candle in the list
        current_idx = len(warm_candles) - 1

        # Ensure the ADX for the current bar is valid (not NaN)
        if np.isnan(adx_values[current_idx]):
            continue

        # 4. Check for strong trend
        is_trending = adx_values[current_idx] > ADX_THRESHOLD

        if is_trending:
            is_sustained_uptrend = True
            for i in range(TREND_CONFIRMATION_LOOKBACK):
                check_idx = current_idx - i
                # Ensure the index is valid and DI values are not NaN
                if check_idx < ADX_PERIOD or np.isnan(plus_di_values[check_idx]) or np.isnan(minus_di_values[check_idx]):
                    is_sustained_uptrend = False
                    break
                if not (plus_di_values[check_idx] > minus_di_values[check_idx]):
                    is_sustained_uptrend = False
                    break
            
            is_sustained_downtrend = True
            # Only check for downtrend if an uptrend was not confirmed to avoid conflicting signals
            if not is_sustained_uptrend:
                for i in range(TREND_CONFIRMATION_LOOKBACK):
                    check_idx = current_idx - i
                    # Ensure the index is valid and DI values are not NaN
                    if check_idx < ADX_PERIOD or np.isnan(plus_di_values[check_idx]) or np.isnan(minus_di_values[check_idx]):
                        is_sustained_downtrend = False
                        break
                    if not (minus_di_values[check_idx] > plus_di_values[check_idx]):
                        is_sustained_downtrend = False
                        break

            # 5c. Emit signals
            if is_sustained_uptrend:
                signals.append(
                    BuySignal(
                        pair=pair,
                        timestamp=warm_candles[current_idx].hour,
                        price=warm_candles[current_idx].close,
                        rule_id="ADX-Filtered Directional Trend Following",
                    )
                )
            elif is_sustained_downtrend:
                signals.append(
                    SellSignal(
                        pair=pair,
                        timestamp=warm_candles[current_idx].hour,
                        price=warm_candles[current_idx].close,
                        rule_id="ADX-Filtered Directional Trend Following",
                    )
                )
    return signals