from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle

# --- Rule Parameters ---
PERIOD_BB = 20          # Bollinger Band period for candles (number of warm candles)
STD_DEV_BB = 2.0        # Standard deviations for Bollinger Bands
PERIOD_VOLUME_MA = 20   # Volume Moving Average period for ticks (number of hot ticks)
VOLUME_MULTIPLIER = 1.5 # Multiplier for volume spike confirmation

# --- Minimum data points required for calculations ---
# We need at least PERIOD_BB warm candles to calculate the Bollinger Bands over a full window.
MIN_CANDLES_BB = PERIOD_BB

# We need at least PERIOD_VOLUME_MA hot ticks to calculate the Volume Moving Average over a full window.
MIN_TICKS_VOLUME = PERIOD_VOLUME_MA

# Unique identifier for this rule, as per the idea_id in the prompt
RULE_ID = "b80e5bb3-fad7-4818-95f8-18b060278e37"

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # 1. Validate sufficient data for Bollinger Bands (warm candles)
        # We need at least MIN_CANDLES_BB warm candles to perform the calculation.
        if len(pair_data.warm) < MIN_CANDLES_BB:
            continue

        # 2. Validate sufficient data for Volume MA and current tick (hot ticks)
        # We need at least MIN_TICKS_VOLUME hot ticks to perform the calculation and get current data.
        if len(pair_data.hot) < MIN_TICKS_VOLUME:
            continue
        
        # --- Bollinger Band Calculations (using WarmCandle close prices) ---
        # Extract close prices from the last `PERIOD_BB` warm candles.
        # This slice ensures we always get `PERIOD_BB` elements because of the preceding length check.
        closes = np.array([c.close for c in pair_data.warm[-PERIOD_BB:]])

        # Calculate Simple Moving Average (mid_band) of the closing prices.
        mid_band = np.mean(closes)
        
        # Calculate Standard Deviation for Bollinger Bands.
        # Using ddof=0 for population standard deviation, which is common in financial applications
        # for rolling window calculations.
        std_dev = np.std(closes, ddof=0)

        # Skip if standard deviation is zero (e.g., all prices are identical), as bands would be meaningless.
        if std_dev == 0:
            continue

        # Calculate upper and lower Bollinger Bands.
        upper_band = mid_band + (std_dev * STD_DEV_BB)
        lower_band = mid_band - (std_dev * STD_DEV_BB)

        # --- Volume Spike Confirmation (using Tick volume_24h data) ---
        # Extract 24-hour rolling volumes from the last `PERIOD_VOLUME_MA` hot ticks.
        # This slice ensures we always get `PERIOD_VOLUME_MA` elements because of the preceding length check.
        volumes_24h = np.array([t.volume_24h for t in pair_data.hot[-PERIOD_VOLUME_MA:]])

        # Calculate the Simple Moving Average of the 24-hour rolling volumes.
        volume_ma = np.mean(volumes_24h)

        # --- Current Data Points for Signal Evaluation ---
        # Get the most recent tick data for current price, volume, and timestamp.
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_volume = current_tick.volume_24h # The latest available 24h rolling volume
        timestamp = current_tick.polled_at

        # --- Generate Signals ---
        # Buy Signal: Price drops below the lower Bollinger Band AND current volume is significantly
        # above its moving average, indicating a high-conviction reversal opportunity.
        if current_price < lower_band and current_volume > (volume_ma * VOLUME_MULTIPLIER):
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID
            ))
        # Sell Signal: Price rises above the upper Bollinger Band AND current volume is significantly
        # above its moving average, indicating a high-conviction reversal opportunity.
        elif current_price > upper_band and current_volume > (volume_ma * VOLUME_MULTIPLIER):
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID
            ))

    return signals