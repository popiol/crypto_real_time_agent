"""Bollinger Band v2: Wider Bands for Stronger Reversals."""
from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, PairData, SellSignal

K = 2.5  # Standard deviation multiplier, increased from 2.0 to 2.5
MIN_CANDLES = 10 # N-period for SMA and SD calculation


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for SMA/SD and at least one hot tick for current price/timestamp.
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES:
            continue

        closes = [c.close for c in pair_data.warm]
        
        # Double-check: although len(pair_data.warm) is checked, ensure closes list itself is valid.
        if len(closes) < MIN_CANDLES:
            continue

        mean = statistics.mean(closes)
        
        # statistics.stdev requires at least 2 data points.
        # Given MIN_CANDLES >= 10, len(closes) will be >= 10, so stdev won't error on count.
        # We still need to handle the case where stdev is 0 (all closes are identical),
        # as this would result in flat bands and no signals.
        std = statistics.stdev(closes)

        if std == 0:
            continue # If standard deviation is zero, bands are flat, no meaningful signal.

        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # Calculate Bollinger Bands with the wider multiplier K
        lower_band = mean - K * std
        upper_band = mean + K * std

        # Generate Buy signal if current price falls below the wider lower band
        if current_price < lower_band:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Generate Sell signal if current price rises above the wider upper band
        elif current_price > upper_band:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals