from __future__ import annotations

import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal

# Rule constants
BOLLINGER_PERIOD = 20
STD_DEV_MULTIPLIER = 2.0
VOLUME_SMA_PERIOD = 20  # Changed from 50 to 20 as per pseudocode
VOLUME_STD_DEV_MULTIPLIER = 1.5 # New constant for volume confirmation

# Minimum number of warm candles required to calculate all indicators
MIN_CANDLES_REQUIRED = max(BOLLINGER_PERIOD, VOLUME_SMA_PERIOD)

# Unique identifier for this rule, derived from the idea_id
RULE_ID = "405493d2-3fca-4288-a744-bacfc4a77397"


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candle data for calculations
        if len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue

        # Extract close prices and volumes from warm candles
        closes = np.array([c.close for c in pair_data.warm])
        volumes = np.array([c.volume for c in pair_data.warm])

        # --- Calculate Bollinger Bands ---
        # We need at least BOLLINGER_PERIOD candles for this
        if len(closes) < BOLLINGER_PERIOD:
            continue

        # Calculate Middle Band (SMA of close prices)
        middle_band = np.mean(closes[-BOLLINGER_PERIOD:])

        # Calculate Standard Deviation of close prices
        std_dev_price = np.std(closes[-BOLLINGER_PERIOD:])

        # Avoid division by zero or nonsensical bands if std_dev is 0
        if std_dev_price == 0:
            continue

        # Calculate Upper and Lower Bollinger Bands
        upper_band = middle_band + (std_dev_price * STD_DEV_MULTIPLIER)
        lower_band = middle_band - (std_dev_price * STD_DEV_MULTIPLIER)

        # --- Calculate Volume SMA and Standard Deviation ---
        # We need at least VOLUME_SMA_PERIOD candles for this
        if len(volumes) < VOLUME_SMA_PERIOD:
            continue

        volume_sma = np.mean(volumes[-VOLUME_SMA_PERIOD:])
        volume_std_dev = np.std(volumes[-VOLUME_SMA_PERIOD:])

        # Calculate the dynamic volume threshold
        # Avoid division by zero or nonsensical threshold if volume_std_dev is 0,
        # in which case the threshold just becomes the SMA.
        volume_threshold = volume_sma + (volume_std_dev * VOLUME_STD_DEV_MULTIPLIER)

        # --- Get Current Market Data ---
        # We need at least one hot tick for the current price and timestamp
        if not pair_data.hot:
            continue

        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # Current volume is taken from the latest warm candle, aligning with the historical volume data
        current_volume = pair_data.warm[-1].volume

        # --- Generate Signals with Enhanced Volume Confirmation ---
        # Buy signal: Price drops below Lower Band AND current volume exceeds the dynamic Volume Threshold
        if current_price < lower_band and current_volume > volume_threshold:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=ts,
                    price=current_price,
                    rule_id=RULE_ID,
                )
            )
        # Sell signal: Price rises above Upper Band AND current volume exceeds the dynamic Volume Threshold
        elif current_price > upper_band and current_volume > volume_threshold:
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=ts,
                    price=current_price,
                    rule_id=RULE_ID,
                )
            )

    return signals