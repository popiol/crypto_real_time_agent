from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Constants for MACD calculation, as per pseudocode
FAST_MA_PERIOD = 12
SLOW_MA_PERIOD = 26
SIGNAL_LINE_PERIOD = 9

# Minimum candles required to calculate MACD and Signal Line for a crossover detection.
# To get the first valid MACD value, we need SLOW_MA_PERIOD candles.
# To get the first valid Signal Line value from MACD, we need SIGNAL_LINE_PERIOD MACD values.
# This means the first valid Signal Line value appears after (SLOW_MA_PERIOD - 1) + (SIGNAL_LINE_PERIOD - 1)
# elements are calculated in the MACD series. The index of this first valid Signal Line value will be
# (SLOW_MA_PERIOD - 1) + (SIGNAL_LINE_PERIOD - 1).
# To detect a crossover, we need at least two consecutive valid MACD and Signal Line values.
# So, we need (SLOW_MA_PERIOD - 1) + (SIGNAL_LINE_PERIOD - 1) + 1 + 1 = SLOW_MA_PERIOD + SIGNAL_LINE_PERIOD
# candles to have two fully formed (non-NaN) MACD and Signal Line pairs.
MIN_CANDLES_REQUIRED = SLOW_MA_PERIOD + SIGNAL_LINE_PERIOD # 26 + 9 = 35

def _calculate_ema_series(data: list[float], period: int) -> list[float]:
    """
    Calculates the Exponential Moving Average (EMA) series for a given list of data.
    The first `period - 1` values will be NaN, and the `period`-th value (at index `period - 1`)
    is initialized using a Simple Moving Average (SMA) of the first `period` data points.
    Subsequent values are calculated using the standard EMA formula.
    Returns a list of EMA values of the same length as the input data.
    """
    if len(data) < period:
        return [np.nan] * len(data)

    ema_values = [np.nan] * len(data)
    
    # Initialize the first EMA value with the SMA of the first 'period' data points
    ema_values[period - 1] = np.mean(data[:period])
    
    multiplier = 2 / (period + 1)
    
    # Calculate subsequent EMAs
    for i in range(period, len(data)):
        ema_values[i] = (data[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        
    return ema_values

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates buy and sell signals based on MACD line and Signal line crossovers.
    A Buy signal is emitted when the MACD line crosses above the Signal line.
    A Sell signal is emitted when the MACD line crosses below the Signal line.
    Requires sufficient warm candle data to calculate the indicators.
    """
    signals: list[BuySignal | SellSignal] = []
    rule_id = "9075a0e8-2b0e-4c3f-8b32-f8874b2cc664" # Unique ID for this rule

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        
        # Ensure we have enough warm candle data for full MACD and Signal Line calculation.
        # The 'warm' data is typically limited (e.g., max 24 entries), which may be insufficient
        # for standard MACD periods (requiring 35 candles for a crossover detection).
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract close prices and corresponding timestamps from warm candles
        close_prices = [candle.close for candle in warm_candles]
        timestamps = [candle.hour for candle in warm_candles]

        # Calculate EMA series for the fast and slow periods
        ema_fast_series = _calculate_ema_series(close_prices, FAST_MA_PERIOD)
        ema_slow_series = _calculate_ema_series(close_prices, SLOW_MA_PERIOD)

        # Calculate the MACD Line (Fast EMA - Slow EMA)
        macd_line_series = []
        for i in range(len(close_prices)):
            if not np.isnan(ema_fast_series[i]) and not np.isnan(ema_slow_series[i]):
                macd_line_series.append(ema_fast_series[i] - ema_slow_series[i])
            else:
                macd_line_series.append(np.nan)

        # Extract only the valid (non-NaN) MACD values to calculate the Signal Line EMA.
        # The first (SLOW_MA_PERIOD - 1) MACD values are NaN; these are skipped.
        temp_macd_for_signal = [m for m in macd_line_series if not np.isnan(m)]
        
        # If there aren't enough valid MACD points to calculate the Signal Line, skip.
        if len(temp_macd_for_signal) < SIGNAL_LINE_PERIOD:
            continue

        # Calculate the Signal Line (EMA of the MACD Line)
        signal_line_temp_series = _calculate_ema_series(temp_macd_for_signal, SIGNAL_LINE_PERIOD)

        # Reconstruct the full signal line series, aligning it with the original timestamps/candles.
        # It needs leading NaNs corresponding to the initial NaNs of the slow EMA,
        # plus any leading NaNs from the signal line EMA itself (which _calculate_ema_series handles).
        num_leading_nans_for_signal_line = (SLOW_MA_PERIOD - 1) # NaNs due to slow EMA's formation
        signal_line_series = [np.nan] * num_leading_nans_for_signal_line
        signal_line_series.extend(signal_line_temp_series)
        
        # Ensure signal_line_series has the same length as warm_candles (pad with NaNs if necessary, though it should match)
        while len(signal_line_series) < len(warm_candles):
            signal_line_series.append(np.nan)


        # Collect only the fully calculated (non-NaN) MACD and Signal Line values
        # along with their corresponding timestamps and prices, for crossover detection.
        valid_indicator_data = []
        for i in range(len(warm_candles)):
            if not np.isnan(macd_line_series[i]) and not np.isnan(signal_line_series[i]):
                valid_indicator_data.append({
                    'macd': macd_line_series[i],
                    'signal': signal_line_series[i],
                    'timestamp': timestamps[i],
                    'price': close_prices[i]
                })
        
        # We need at least two valid data points (current and previous) to detect a crossover.
        if len(valid_indicator_data) < 2:
            continue

        # Get the latest and previous fully calculated indicator values
        latest_data = valid_indicator_data[-1]
        previous_data = valid_indicator_data[-2]

        # MACD Crossover Logic
        # Bullish crossover: MACD crosses above Signal Line
        if latest_data['macd'] > latest_data['signal'] and previous_data['macd'] <= previous_data['signal']:
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_data['timestamp'],
                price=latest_data['price'],
                rule_id=rule_id
            ))
        # Bearish crossover: MACD crosses below Signal Line
        elif latest_data['macd'] < latest_data['signal'] and previous_data['macd'] >= previous_data['signal']:
            signals.append(SellSignal(
                pair=pair,
                timestamp=latest_data['timestamp'],
                price=latest_data['price'],
                rule_id=rule_id
            ))

    return signals