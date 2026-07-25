from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

# Constants for the SMA periods
SHORT_MA_PERIOD = 8
LONG_MA_PERIOD = 20

# Rule ID as specified in the idea
RULE_ID = "SMA-Crossover-Refinement-001"

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
        # Use data.warm as the source for hourly candles.
        # This aligns with the rule's rationale for "short-term trend changes"
        # and the "hourly candles" mentioned in the description, despite
        # the pseudocode's likely typo `data.cold.close` (as data.cold contains
        # monthly aggregates and no 'close' attribute).
        warm_data: list[WarmCandle] = pair_data.warm

        # Ensure enough data for the longest MA to have at least two values
        # for crossover detection. To get N SMA values, we need (N + PERIOD - 1) prices.
        # For 2 SMA values (current and previous), we need (2 + LONG_MA_PERIOD - 1)
        # = LONG_MA_PERIOD + 1 prices.
        # With LONG_MA_PERIOD = 20, we need 21 warm candles.
        # `data.warm` holds at most 24 hourly candles, so this is feasible.
        required_data_points = LONG_MA_PERIOD + 1
        if len(warm_data) < required_data_points:
            # Not enough hourly data to calculate the long MA and its previous value
            continue

        # Extract 'close' prices from the chronologically ordered WarmCandle objects.
        prices = [candle.close for candle in warm_data]

        short_ma_series = _calculate_sma(prices, SHORT_MA_PERIOD)
        long_ma_series = _calculate_sma(prices, LONG_MA_PERIOD)

        # After calculation, ensure both MA series have at least 2 values.
        # This provides a safeguard if `_calculate_sma` returns an empty list
        # due to edge cases with period calculation, or if `required_data_points`
        # logic was somehow insufficient for both series (though it should be fine).
        if len(short_ma_series) < 2 or len(long_ma_series) < 2:
            continue

        # Get current and previous SMA values from the end of each series
        current_short_ma = short_ma_series[-1]
        previous_short_ma = short_ma_series[-2]
        current_long_ma = long_ma_series[-1]
        previous_long_ma = long_ma_series[-2]

        # Determine the most recent price and timestamp for the signal.
        # Signals should reflect current market conditions.
        current_price: float | None = None
        timestamp: datetime | None = None

        if pair_data.hot:
            # Use the latest tick data if available
            last_tick: Tick = pair_data.hot[-1]
            current_price = last_tick.last_price
            timestamp = last_tick.polled_at
        elif pair_data.warm:
            # Otherwise, use the latest warm candle data
            last_candle: WarmCandle = pair_data.warm[-1]
            current_price = last_candle.close
            timestamp = last_candle.hour
        else:
            # No current price available from hot or warm data. Cannot generate a relevant signal.
            continue
        
        # Ensure current_price and timestamp are not None before creating signals
        if current_price is None or timestamp is None:
            continue

        # Determine trading signal based on crossover logic
        # Buy Signal: Short MA (8) crosses above Long MA (20)
        # Pseudocode: IF previous_SMA_short < previous_SMA_long AND current_SMA_short > current_SMA_long THEN
        if previous_short_ma < previous_long_ma and current_short_ma > current_long_ma:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID
            ))
        # Sell Signal: Short MA (8) crosses below Long MA (20)
        # Pseudocode: ELSE IF previous_SMA_short > previous_SMA_long AND current_SMA_short < current_SMA_long THEN
        elif previous_short_ma > previous_long_ma and current_short_ma < current_long_ma:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID
            ))

    return signals