"""Rule 03 — Bollinger Band V3: Asymmetric Bands."""
from __future__ import annotations

import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# Default asymmetric K multipliers
K_BUY = 1.8  # Multiplier for the lower band (buy signal sensitivity)
K_SELL = 2.2 # Multiplier for the upper band (sell signal sensitivity)

MIN_CANDLES = 10 # Minimum number of warm candles required for calculation


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough historical candles and at least one hot tick for current price
        if len(pair_data.warm) < MIN_CANDLES or not pair_data.hot:
            continue

        # Extract closing prices from warm candles
        closes = [c.close for c in pair_data.warm]

        # Calculate Moving Average (MA)
        mean = statistics.mean(closes)

        # Calculate Standard Deviation (STD)
        # stdev requires at least two data points. If MIN_CANDLES is 1, this would fail.
        # With MIN_CANDLES = 10, this is safe.
        std = statistics.stdev(closes)

        # If standard deviation is zero, prices haven't moved, so no band deviation possible.
        if std == 0:
            continue

        # Get current price and timestamp from the most recent tick
        current_tick: Tick = pair_data.hot[-1]
        current_price: float = current_tick.last_price
        ts: datetime = current_tick.polled_at

        # Calculate asymmetric Bollinger Bands
        lower_band_asymmetric = mean - K_BUY * std
        upper_band_asymmetric = mean + K_SELL * std

        # Generate signals based on current price relative to asymmetric bands
        if current_price < lower_band_asymmetric:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif current_price > upper_band_asymmetric:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals