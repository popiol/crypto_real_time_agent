from __future__ import annotations
import numpy as np
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle

# Parameters as per the rule description and pseudocode
WINDOW_BB = 20              # Bollinger Band window
STD_DEV_MULTIPLIER = 2.0    # Standard deviation multiplier for BB
WINDOW_SMA_TREND = 100      # Long-term SMA window for trend filter

# Minimum number of warm candles required to calculate both Bollinger Bands
# and the long-term SMA for the trend filter.
MIN_CANDLES = max(WINDOW_BB, WINDOW_SMA_TREND)

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have at least one hot tick for the current price and timestamp,
        # and enough warm candles to perform all required calculations.
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES:
            continue

        # Extract close prices from the warm candles.
        # These are used for all indicator calculations.
        closes = np.array([c.close for c in pair_data.warm])

        # Get the current price and timestamp from the latest hot tick.
        current_tick: Tick = pair_data.hot[-1]
        current_price: float = current_tick.last_price
        ts: datetime = current_tick.polled_at

        # --- Calculate Bollinger Bands ---
        # The `MIN_CANDLES` check guarantees there are enough `closes` for `WINDOW_BB`.
        bb_closes = closes[-WINDOW_BB:]
        sma_bb = np.mean(bb_closes)
        std_dev_bb = np.std(bb_closes)

        # Skip if standard deviation is zero to avoid division by zero or nonsensical bands.
        # This can happen if all prices in the window are identical.
        if std_dev_bb == 0:
            continue

        upper_band = sma_bb + (STD_DEV_MULTIPLIER * std_dev_bb)
        lower_band = sma_bb - (STD_DEV_MULTIPLIER * std_dev_bb)

        # --- Calculate Trend Filter SMA ---
        # The `MIN_CANDLES` check guarantees there are enough `closes` for `WINDOW_SMA_TREND`.
        sma_trend = np.mean(closes[-WINDOW_SMA_TREND:])

        # --- Generate Signals based on Bollinger Bands and SMA Trend Filter ---

        # Buy Signal:
        # Price drops below the lower Bollinger Band (oversold condition)
        # AND the current price is above its long-term SMA (indicating an uptrend).
        # This suggests a pullback in an uptrend.
        if current_price < lower_band and current_price > sma_trend:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))

        # Sell Signal:
        # Price rises above the upper Bollinger Band (overbought condition)
        # AND the current price is below its long-term SMA (indicating a downtrend).
        # This suggests a bounce in a downtrend.
        elif current_price > upper_band and current_price < sma_trend:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals