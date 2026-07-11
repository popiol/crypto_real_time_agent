from __future__ import annotations
from datetime import datetime
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule Constants ---
RULE_ID = "2f6b6270-1970-4e42-8e66-9a2c7c4249d2"  # Unique ID for this rule idea
ADX_PERIOD = 14  # Period for ADX calculation, as specified in the pseudocode
ADX_THRESHOLD = 20  # Threshold for ADX to indicate a strong trend, as specified
EMA_PERIOD = 9  # Period for the Exponential Moving Average, as specified

# Minimum candles needed to calculate all indicators for the latest candle
# and have a previous candle for crossover checks.
# ADX requires 2 * ADX_PERIOD - 1 candles for its first valid value (index 2*ADX_PERIOD - 1).
# EMA requires EMA_PERIOD - 1 candles for its first valid value (index EMA_PERIOD - 1).
# To have valid values for the current candle (index -1) and previous candle (index -2) for crossover:
# We need `len(candles) - 1 >= max(2 * ADX_PERIOD - 1, EMA_PERIOD - 1)`
# And `len(candles) - 2 >= max(ADX_PERIOD - 1, EMA_PERIOD - 1)` (for DI crossover's previous values)
# Therefore, `len(candles)` must be at least `max(2 * ADX_PERIOD - 1, EMA_PERIOD - 1) + 2`.
MIN_CANDLES_FOR_SIGNAL = max(2 * ADX_PERIOD - 1, EMA_PERIOD - 1) + 2


def _wilders_smoothing(data: list[float], period: int) -> list[float]:
    """
    Calculates Wilder's Smoothing (RMA) for a given data series.
    The first 'period-1' elements of the returned list will be 0.0,
    and the actual smoothed values start from index 'period-1'.
    The first valid smoothed value is the simple average of the first 'period' raw values.
    Subsequent values use the Wilder's smoothing formula:
    Smoothed_t = (Smoothed_{t-1} * (period - 1) + Current_Value) / period
    """
    if not data or len(data) < period:
        return [0.0] * len(data)

    smoothed = [0.0] * len(data)

    # The first smoothed value (at index period-1) is the simple average of the first 'period' values
    smoothed[period - 1] = sum(data[:period]) / period

    # Apply Wilder's smoothing for subsequent values
    for i in range(period, len(data)):
        smoothed[i] = (smoothed[i - 1] * (period - 1) + data[i]) / period

    return smoothed


def calculate_adx(
    candles: list[WarmCandle], period: int
) -> tuple[list[float], list[float], list[float]]:
    """
    Calculates ADX, +DI, and -DI for a list of WarmCandle objects.
    Returns three lists: ADX values, +DI values, -DI values.
    Each list will contain leading zeros for periods where ADX/DI is not yet available,
    consistent with the _wilders_smoothing function's output.
    """
    num_candles = len(candles)
    if num_candles < 2:  # Need at least 2 candles for TR, DM calculations
        return [], [], []

    # 1. Prepare OHLC data
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]

    # 2. Calculate True Range (TR)
    tr_values = [0.0] * num_candles
    # The first TR value is simply High - Low for the first candle
    tr_values[0] = highs[0] - lows[0]

    for i in range(1, num_candles):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_values[i] = max(hl, hc, lc)

    # 3. Calculate Directional Movement (+DM, -DM)
    pdm_values = [0.0] * num_candles  # Positive Directional Movement
    ndm_values = [0.0] * num_candles  # Negative Directional Movement

    for i in range(1, num_candles):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        if up_move > down_move and up_move > 0:
            pdm_values[i] = up_move
            ndm_values[i] = 0.0
        elif down_move > up_move and down_move > 0:
            pdm_values[i] = 0.0
            ndm_values[i] = down_move
        else:
            pdm_values[i] = 0.0
            ndm_values[i] = 0.0

    # 4. Smooth TR, +DM, -DM using Wilder's Smoothing (RMA)
    smoothed_tr = _wilders_smoothing(tr_values, period)
    smoothed_pdm = _wilders_smoothing(pdm_values, period)
    smoothed_ndm = _wilders_smoothing(ndm_values, period)

    # 5. Calculate Directional Indicators (+DI, -DI)
    di_plus = [0.0] * num_candles
    di_minus = [0.0] * num_candles

    for i in range(num_candles):
        if smoothed_tr[i] > 0:  # Avoid division by zero
            di_plus[i] = (smoothed_pdm[i] / smoothed_tr[i]) * 100
            di_minus[i] = (smoothed_ndm[i] / smoothed_tr[i]) * 100

    # 6. Calculate Directional Index (DX)
    dx_values = [0.0] * num_candles
    for i in range(num_candles):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0:  # Avoid division by zero
            dx_values[i] = abs((di_plus[i] - di_minus[i]) / di_sum) * 100

    # 7. Smooth DX to get ADX
    # ADX is the Wilder's smoothed version of DX, typically over the same period.
    adx_values = _wilders_smoothing(dx_values, period)

    return adx_values, di_plus, di_minus


def calculate_ema(prices: list[float], period: int) -> list[float]:
    """
    Calculates Exponential Moving Average (EMA) for a given price series.
    The first 'period-1' elements of the returned list will be 0.0,
    and the actual EMA values start from index 'period-1'.
    The first valid EMA value is typically the Simple Moving Average (SMA)
    of the first 'period' values.
    """
    if not prices or len(prices) < period:
        return [0.0] * len(prices)

    ema_values = [0.0] * len(prices)
    
    # Calculate the initial EMA (SMA for the first 'period' values)
    # This value is stored at index `period - 1`
    ema_values[period - 1] = sum(prices[:period]) / period
    
    multiplier = 2 / (period + 1)
    
    # Apply EMA formula for subsequent values
    for i in range(period, len(prices)):
        ema_values[i] = (prices[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
        
    return ema_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates Buy or Sell signals based on ADX, +DI/-DI crossovers, and an EMA price filter.
    A Buy signal is generated when ADX indicates a strong trend (+DI crosses above -DI)
    AND the current price is above the short-term EMA.
    A Sell signal is generated when ADX indicates a strong trend (-DI crosses above +DI)
    AND the current price is below the short-term EMA.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # Ensure enough historical data for all calculations and crossover checks
        if len(candles) < MIN_CANDLES_FOR_SIGNAL:
            continue

        closes = [c.close for c in candles]

        # Calculate ADX, +DI, -DI
        adx_values, di_plus, di_minus = calculate_adx(candles, ADX_PERIOD)
        # Calculate EMA
        ema_values = calculate_ema(closes, EMA_PERIOD)

        # Get the latest and previous indicator values for the current candle (index -1)
        # and the candle before it (index -2).
        # These indices are guaranteed to be valid due to MIN_CANDLES_FOR_SIGNAL check.
        current_adx = adx_values[-1]
        current_di_plus = di_plus[-1]
        current_di_minus = di_minus[-1]
        current_ema = ema_values[-1]
        current_close_price = closes[-1]

        previous_di_plus = di_plus[-2]
        previous_di_minus = di_minus[-2]

        # Check ADX threshold for trend strength
        if current_adx > ADX_THRESHOLD:
            # BUY signal conditions:
            # 1. ADX indicates strong trend (current_adx > ADX_THRESHOLD)
            # 2. +DI crosses above -DI (current +DI > current -DI AND previous +DI <= previous -DI)
            # 3. Current close price is above EMA
            if (
                current_di_plus > current_di_minus
                and previous_di_plus <= previous_di_minus
                and current_close_price > current_ema
            ):
                signals.append(
                    BuySignal(
                        pair=pair,
                        timestamp=candles[-1].hour,
                        price=current_close_price,
                        rule_id=RULE_ID,
                        confidence=current_adx / 100.0,  # ADX strength as confidence (0-1)
                    )
                )
            # SELL signal conditions:
            # 1. ADX indicates strong trend (current_adx > ADX_THRESHOLD)
            # 2. -DI crosses above +DI (current -DI > current +DI AND previous -DI <= previous +DI)
            # 3. Current close price is below EMA
            elif (
                current_di_minus > current_di_plus
                and previous_di_minus <= previous_di_plus
                and current_close_price < current_ema
            ):
                signals.append(
                    SellSignal(
                        pair=pair,
                        timestamp=candles[-1].hour,
                        price=current_close_price,
                        rule_id=RULE_ID,
                        confidence=current_adx / 100.0,  # ADX strength as confidence (0-1)
                    )
                )

    return signals