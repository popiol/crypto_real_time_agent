from __future__ import annotations
from datetime import datetime
import numpy as np  # Required by the problem statement, though not strictly necessary for simple math.
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule Constants ---
RULE_ID = "81609980-0e71-4197-a0e8-a73337a8a753"
ADX_PERIOD = 10  # Common period is 14, but reduced to 10 to allow signals with max 24 warm candles.
ADX_THRESHOLD = 25.0
# Minimum candles needed for the first valid ADX value using standard Wilder's smoothing:
# PERIOD candles for initial TR/DM smoothing, then another PERIOD candles for DX smoothing.
MIN_CANDLES_FOR_ADX = 2 * ADX_PERIOD


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


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates Buy or Sell signals based on the Average Directional Index (ADX)
    and its directional indicators (+DI and -DI).
    A Buy signal is generated when ADX is above a threshold and +DI crosses above -DI.
    A Sell signal is generated when ADX is above a threshold and -DI crosses above +DI.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # Ensure enough historical data for ADX calculation
        if len(candles) < MIN_CANDLES_FOR_ADX:
            continue

        adx_values, di_plus, di_minus = calculate_adx(candles, ADX_PERIOD)

        # We need the most recent ADX, +DI, -DI values, and their previous values for crossover check.
        # The valid calculated ADX/DI values start from index MIN_CANDLES_FOR_ADX - 1.
        # The latest valid values are at `len(candles) - 1`.
        # For a crossover, we need the current and the immediately preceding valid values.
        if len(adx_values) < MIN_CANDLES_FOR_ADX:
            continue # Should not happen if the initial check `len(candles) < MIN_CANDLES_FOR_ADX` passed, but good for safety.

        current_adx = adx_values[-1]
        current_di_plus = di_plus[-1]
        current_di_minus = di_minus[-1]

        # Ensure we have a previous valid point for crossover check.
        if len(adx_values) < MIN_CANDLES_FOR_ADX + 1:
            continue

        previous_di_plus = di_plus[-2]
        previous_di_minus = di_minus[-2]

        # Check ADX threshold for trend strength
        if current_adx > ADX_THRESHOLD:
            # BUY signal: ADX > threshold AND +DI > -DI AND +DI crossed above -DI
            if (current_di_plus > current_di_minus) and (
                previous_di_plus <= previous_di_minus
            ):
                signals.append(
                    BuySignal(
                        pair=pair,
                        timestamp=candles[-1].hour,  # Use the timestamp of the last candle
                        price=candles[-1].close,
                        rule_id=RULE_ID,
                        confidence=current_adx / 100.0,  # ADX strength as confidence (0-1)
                    )
                )
            # SELL signal: ADX > threshold AND -DI > +DI AND -DI crossed above +DI
            elif (current_di_minus > current_di_plus) and (
                previous_di_minus <= previous_di_plus
            ):
                signals.append(
                    SellSignal(
                        pair=pair,
                        timestamp=candles[-1].hour,
                        price=candles[-1].close,
                        rule_id=RULE_ID,
                        confidence=current_adx / 100.0,
                    )
                )

    return signals