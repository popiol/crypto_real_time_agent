from __future__ import annotations

import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, Tick
from datetime import datetime

# --- Constants ---
# Bollinger Band period (for SMA and Standard Deviation)
BB_PERIOD = 20
# Bollinger Band standard deviation multiplier
BB_STD_DEV = 2.0
# Long-term SMA period for trend filtering
# Note: `pair_data.warm` contains at most 24 hourly candles.
# This value must be <= 24 to be calculated from `warm` data.
SMA_PERIOD_LONGTERM = 24

# Minimum number of warm candles required to perform calculations.
# This must be at least the maximum of the BB_PERIOD and SMA_PERIOD_LONGTERM.
MIN_CANDLES_REQUIRED = max(BB_PERIOD, SMA_PERIOD_LONGTERM)

# --- Helper Functions (using numpy for efficiency) ---
def _calculate_sma(data: np.ndarray, period: int) -> float:
    """Calculates the Simple Moving Average for the last 'period' elements."""
    if len(data) < period:
        raise ValueError(f"Not enough data ({len(data)}) to calculate SMA for period {period}.")
    return np.mean(data[-period:])

def _calculate_std_dev(data: np.ndarray, period: int) -> float:
    """Calculates the Standard Deviation for the last 'period' elements."""
    if len(data) < period:
        raise ValueError(f"Not enough data ({len(data)}) to calculate Standard Deviation for period {period}.")
    # numpy's std uses N by default (population std dev), which is common for BB.
    return np.std(data[-period:])

# --- Main Signal Function ---
def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # 1. Data Sufficiency Check
        # Ensure enough warm candles are available for all calculations
        if len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue
        # Ensure hot data is available for current price and timestamp
        if not pair_data.hot:
            continue

        # Extract close prices from the warm candles
        # We need the last `MIN_CANDLES_REQUIRED` closes
        closes_warm = np.array([c.close for c in pair_data.warm])

        # 2. Calculate Bollinger Bands
        try:
            middle_band = _calculate_sma(closes_warm, BB_PERIOD)
            std_dev = _calculate_std_dev(closes_warm, BB_PERIOD)
        except ValueError:
            # Should not happen if MIN_CANDLES_REQUIRED check is sufficient,
            # but added for robustness.
            continue

        # If standard deviation is zero, bands are meaningless; skip.
        if std_dev == 0:
            continue

        upper_band = middle_band + (std_dev * BB_STD_DEV)
        lower_band = middle_band - (std_dev * BB_STD_DEV)

        # 3. Calculate Long-term SMA for trend filtering
        try:
            long_term_sma = _calculate_sma(closes_warm, SMA_PERIOD_LONGTERM)
        except ValueError:
            # Should not happen if MIN_CANDLES_REQUIRED check is sufficient.
            continue

        # 4. Get latest price and timestamp from hot data
        latest_tick: Tick = pair_data.hot[-1]
        current_price = latest_tick.last_price
        ts = latest_tick.polled_at

        # 5. Apply Trading Rule Logic
        # Buy signal: price below lower BB AND current price is above long-term SMA (uptrend)
        if current_price < lower_band and current_price > long_term_sma:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Sell signal: price above upper BB AND current price is below long-term SMA (downtrend)
        elif current_price > upper_band and current_price < long_term_sma:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals