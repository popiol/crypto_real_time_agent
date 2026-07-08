from __future__ import annotations

import statistics

from src.agent.models import BuySignal, MarketData, SellSignal

# Parameters for Bollinger Bands
BB_LENGTH = 20
BB_K = 2.0  # Standard deviation multiplier

# Parameter for the Trend Moving Average
TREND_MA_LENGTH = 100

# Minimum number of warm candles required to calculate all indicators
MIN_CANDLES = max(BB_LENGTH, TREND_MA_LENGTH)


def _calculate_sma(data: list[float], length: int) -> float:
    """Calculates the Simple Moving Average for the last 'length' elements of data."""
    if len(data) < length:
        raise ValueError(f"Not enough data ({len(data)}) for SMA of length {length}")
    return statistics.mean(data[-length:])


def _calculate_stddev(data: list[float], length: int) -> float:
    """Calculates the Standard Deviation for the last 'length' elements of data."""
    if len(data) < length:
        raise ValueError(f"Not enough data ({len(data)}) for STDDEV of length {length}")
    # statistics.stdev requires at least 2 data points.
    if length < 2:
        return 0.0
    # If all values are identical, stdev is 0. This is handled correctly by statistics.stdev
    return statistics.stdev(data[-length:])


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the "Bollinger Band with Trend Confirmation" trading rule.

    This rule emits a Buy signal when the price falls below the lower Bollinger Band
    AND the current price is above a longer-period Simple Moving Average, indicating
    an oversold condition within an uptrend.

    It emits a Sell signal when the price rises above the upper Bollinger Band
    AND the current price is below a longer-period Simple Moving Average, indicating
    an overbought condition within a downtrend.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm (hourly candle) data for calculations
        if len(pair_data.warm) < MIN_CANDLES:
            continue

        # Ensure we have at least one hot (real-time tick) data point for current price and timestamp
        if not pair_data.hot:
            continue

        closes = [c.close for c in pair_data.warm]

        # Calculate Bollinger Bands
        sma_bb = _calculate_sma(closes, BB_LENGTH)
        stddev_bb = _calculate_stddev(closes, BB_LENGTH)

        # If standard deviation is zero, bands collapse to the SMA.
        # This means price would have to be exactly on the SMA to trigger,
        # which is extremely unlikely. We can continue or let it play out.
        # For simplicity, we'll let it play out, as the conditions will likely not be met.
        # The pseudocode does not explicitly skip this case.

        upper_band = sma_bb + (BB_K * stddev_bb)
        lower_band = sma_bb - (BB_K * stddev_bb)

        # Calculate Trend Moving Average
        trend_ma = _calculate_sma(closes, TREND_MA_LENGTH)

        # Get current price and timestamp from the most recent tick
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        ts = current_tick.polled_at

        # Generate Signals based on the rule logic
        # Buy signal: Price is oversold (below lower BB) AND above long-term trend (uptrend)
        if current_price < lower_band and current_price > trend_ma:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Sell signal: Price is overbought (above upper BB) AND below long-term trend (downtrend)
        elif current_price > upper_band and current_price < trend_ma:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals