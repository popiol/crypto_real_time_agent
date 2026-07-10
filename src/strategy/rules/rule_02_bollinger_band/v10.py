from __future__ import annotations

import numpy as np

from src.agent.models import BuySignal, MarketData, SellSignal


# Parameters
N_BB = 20  # Bollinger Band period (number of ticks)
K_BB = 2.0  # Bollinger Band standard deviation multiplier
N_MA_Trend = 200  # Long-term Moving Average period for trend filter (number of ticks)

# Minimum data points required for calculations.
# This must be at least the largest period used (N_MA_Trend).
MIN_TICKS = max(N_BB, N_MA_Trend)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements a Bollinger Band Mean Reversion strategy with a trend filter.

    It emits a Buy signal when the price drops below the lower Bollinger Band
    AND the longer-term trend (defined by a N_MA_Trend-period SMA) is upward
    (current price > LongTermMA).
    It emits a Sell signal when the price rises above the upper Bollinger Band
    AND the longer-term trend is downward (current price < LongTermMA).

    This rule uses `last_price` from `hot` (tick) data for calculations to
    accommodate the `N_MA_Trend = 200` period, as `warm` (hourly candle) data
    typically does not provide enough historical depth for such a long period.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough tick data for both BB and trend MA calculations
        if not pair_data.hot or len(pair_data.hot) < MIN_TICKS:
            continue

        # Extract closing prices (last_price) from the hot (tick) data
        # We need the most recent `MIN_TICKS` for calculations.
        closes = np.array([t.last_price for t in pair_data.hot[-MIN_TICKS:]], dtype=float)
        current_price = closes[-1]
        ts = pair_data.hot[-1].polled_at

        # Calculate Bollinger Bands (N_BB period)
        # We use the last N_BB prices for the Bollinger Band calculations.
        bb_segment = closes[-N_BB:]
        bb_mean = np.mean(bb_segment)
        bb_std = np.std(bb_segment)

        # Avoid division by zero or nonsensical bands if std dev is zero
        if bb_std == 0:
            continue

        upper_band = bb_mean + (K_BB * bb_std)
        lower_band = bb_mean - (K_BB * bb_std)

        # Calculate Long-term Trend MA (N_MA_Trend period)
        # We use the last N_MA_Trend prices for the Long-term MA.
        trend_ma_segment = closes[-N_MA_Trend:]
        long_term_ma = np.mean(trend_ma_segment)

        # Generate Signals based on Bollinger Bands and Trend Filter
        # Buy Signal: Price below lower band AND current price is above long-term MA (upward trend)
        if current_price < lower_band and current_price > long_term_ma:
            signals.append(
                BuySignal(pair=pair, timestamp=ts, price=current_price)
            )
        # Sell Signal: Price above upper band AND current price is below long-term MA (downward trend)
        elif current_price > upper_band and current_price < long_term_ma:
            signals.append(
                SellSignal(pair=pair, timestamp=ts, price=current_price)
            )

    return signals