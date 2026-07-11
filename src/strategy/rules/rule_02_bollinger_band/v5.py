"""Rule 02 — Bollinger Band Breach with RSI Confirmation (v1 enhancement)."""
from __future__ import annotations

import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal

# Bollinger Band Parameters
BB_PERIOD = 20
BB_STD_DEV = 2.0

# RSI Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Minimum number of candles required for calculations
# BB needs BB_PERIOD candles.
# RSI needs RSI_PERIOD + 1 candles (for 'period' differences).
MIN_CANDLES = max(BB_PERIOD, RSI_PERIOD + 1)


def _calculate_bollinger_bands(
    prices: np.ndarray, period: int, num_std_dev: float
) -> tuple[float, float, float]:
    """Calculates SMA, Upper Bollinger Band, and Lower Bollinger Band."""
    if len(prices) < period:
        return np.nan, np.nan, np.nan

    # Use the last 'period' prices for calculation
    relevant_prices = prices[-period:]
    sma = np.mean(relevant_prices)
    std_dev = np.std(relevant_prices)

    upper_band = sma + (num_std_dev * std_dev)
    lower_band = sma - (num_std_dev * std_dev)
    return sma, upper_band, lower_band


def _calculate_rsi(prices: np.ndarray, period: int) -> float:
    """Calculates the Relative Strength Index (RSI) for the last price in the series."""
    if len(prices) < period + 1:
        return np.nan

    # Calculate price differences (diffs[i] = prices[i+1] - prices[i])
    diffs = np.diff(prices)

    # Separate gains and losses
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)

    # Use the last 'period' differences for average gain/loss calculation
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0  # If no losses, RSI is 100 (or 50 if no change)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for calculations
        if not pair_data.warm or len(pair_data.warm) < MIN_CANDLES:
            continue

        # Extract close prices from warm candles
        closes = np.array([c.close for c in pair_data.warm], dtype=np.float64)

        # Calculate Bollinger Bands
        _, upper_bb, lower_bb = _calculate_bollinger_bands(
            closes, BB_PERIOD, BB_STD_DEV
        )

        # Calculate RSI
        rsi = _calculate_rsi(closes, RSI_PERIOD)

        # Check for insufficient data from helper functions (e.g., if MIN_CANDLES was miscalculated)
        if np.isnan(upper_bb) or np.isnan(lower_bb) or np.isnan(rsi):
            continue

        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        timestamp = current_tick.polled_at

        # Generate signals based on the rule
        if current_price < lower_bb and rsi < RSI_OVERSOLD:
            signals.append(BuySignal(pair=pair, timestamp=timestamp, price=current_price))
        elif current_price > upper_bb and rsi > RSI_OVERBOUGHT:
            signals.append(SellSignal(pair=pair, timestamp=timestamp, price=current_price))

    return signals