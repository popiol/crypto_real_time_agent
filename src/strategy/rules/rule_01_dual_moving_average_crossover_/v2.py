from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal

# Constants from the pseudocode
SHORT_MA_PERIOD = 10
LONG_MA_PERIOD = 30

def _calculate_sma(prices: list[float], period: int) -> list[float]:
    """
    Calculates Simple Moving Average using numpy.
    The output SMA values correspond to the end of each 'period' window.
    For example, if prices = [p1, p2, p3, p4, p5] and period = 3,
    the output will be [(p1+p2+p3)/3, (p2+p3+p4)/3, (p3+p4+p5)/3].
    """
    if len(prices) < period:
        return []
    # Using np.convolve for a rolling sum, then divide by period
    # 'valid' mode means output is only where the convolution fully overlaps.
    # The result length will be len(prices) - period + 1.
    sma = np.convolve(prices, np.ones(period), 'valid') / period
    return sma.tolist()

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Use data.warm as specified in the pseudocode
        warm_data = pair_data.warm

        # Ensure enough data for the longest MA to have at least two values
        # for crossover detection.
        # An SMA series of length L requires (L + PERIOD - 1) prices.
        # To get 2 SMA values, we need (2 + PERIOD - 1) = PERIOD + 1 prices.
        # For LONG_MA_PERIOD = 30, we would need 31 warm candles.
        # Given the constraint that `data.warm` holds at most 24 hourly candles,
        # this rule will likely never generate signals as `len(warm_data)`
        # will almost always be less than `required_data_points`.
        required_data_points = LONG_MA_PERIOD + 1
        if len(warm_data) < required_data_points:
            # Not enough hourly data to calculate long MA and its previous value
            continue

        # Extract 'close' prices from each WarmCandle object.
        # These are assumed to be chronologically ordered.
        prices = [candle.close for candle in warm_data]

        short_ma_series = _calculate_sma(prices, SHORT_MA_PERIOD)
        long_ma_series = _calculate_sma(prices, LONG_MA_PERIOD)

        # After calculation, ensure both MA series have at least 2 values.
        # This check covers cases where `_calculate_sma` returned an empty list
        # due to insufficient data for a specific period.
        # Given `data.warm` constraints and `LONG_MA_PERIOD = 30`, `long_ma_series`
        # will almost always have fewer than 2 elements.
        if len(short_ma_series) < 2 or len(long_ma_series) < 2:
            continue

        # Get current and previous SMA values from the end of the series
        current_short_ma = short_ma_series[-1]
        previous_short_ma = short_ma_series[-2]
        current_long_ma = long_ma_series[-1]
        previous_long_ma = long_ma_series[-2]

        # Determine the most recent price and timestamp for the signal.
        # Signals should be executed at current market conditions, not historical averages.
        current_price = None
        timestamp = None

        if pair_data.hot:
            current_price = pair_data.hot[-1].last_price
            timestamp = pair_data.hot[-1].polled_at
        elif pair_data.warm:
            current_price = pair_data.warm[-1].close
            timestamp = pair_data.warm[-1].hour
        else:
            # No current price available from hot or warm data. Cannot generate a relevant signal.
            continue

        # Determine trading signal based on crossover logic
        # Golden Cross: Short MA crosses above Long MA
        if previous_short_ma <= previous_long_ma and current_short_ma > current_long_ma:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id="SMA-PERIOD-ADJ-001"
            ))
        # Death Cross: Short MA crosses below Long MA
        elif previous_short_ma >= previous_long_ma and current_short_ma < current_long_ma:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id="SMA-PERIOD-ADJ-001"
            ))

    return signals