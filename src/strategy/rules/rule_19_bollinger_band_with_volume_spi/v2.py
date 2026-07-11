from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Parameters for Bollinger Bands and Volume Confirmation
PERIOD = 20  # Period for Simple Moving Average and Standard Deviation of close prices
STD_DEV_MULTIPLIER = 2  # Standard deviation multiplier for Bollinger Bands (as per pseudocode)
VOLUME_SMA_PERIOD = 20  # Period for Simple Moving Average of Volume (as per pseudocode)
VOLUME_MULTIPLIER = 1.5  # Multiplier for volume confirmation threshold

# Unique identifier for this trading rule
RULE_ID = "6a2204e9-7889-4ea8-b390-e69c71cfcdb4"

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on Bollinger Bands with Adaptive Volume Confirmation.

    This rule detects extreme price movements outside Bollinger Bands, confirmed by
    current volume exceeding a multiple of its recent average.

    A Buy signal is emitted when the price closes below the lower band and current volume
    is above 1.5 times the 20-period Simple Moving Average of Volume (SMA_Volume),
    indicating an oversold condition with strong conviction.

    A Sell signal is emitted when the price closes above the upper band and current volume
    is above 1.5 times the 20-period SMA_Volume, suggesting an overbought condition
    with strong selling pressure.
    """
    signals: list[BuySignal | SellSignal] = []

    # Determine the minimum number of candles required for both Bollinger Bands
    # and Volume SMA calculations.
    min_required_candles = max(PERIOD, VOLUME_SMA_PERIOD)

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # Ensure enough candles are available for all calculations
        if len(candles) < min_required_candles:
            continue

        # Extract close prices and volumes from the candles.
        # We need enough historical data for the specified periods.
        close_prices = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Calculate Bollinger Bands for the latest point
        # The SMA and STDDEV are calculated on the last 'PERIOD' close prices.
        sma_close = np.mean(close_prices[-PERIOD:])
        std_dev_close = np.std(close_prices[-PERIOD:])
        upper_band = sma_close + (std_dev_close * STD_DEV_MULTIPLIER)
        lower_band = sma_close - (std_dev_close * STD_DEV_MULTIPLIER)

        # Calculate Simple Moving Average of Volume for the latest point
        # The SMA is calculated on the last 'VOLUME_SMA_PERIOD' volumes.
        volume_sma = np.mean(volumes[-VOLUME_SMA_PERIOD:])

        # Get the latest candle's data for signal generation
        latest_candle: WarmCandle = candles[-1]
        current_close = latest_candle.close
        current_volume = latest_candle.volume
        current_timestamp = latest_candle.hour

        # Calculate the adaptive volume confirmation threshold
        volume_confirmation_threshold = volume_sma * VOLUME_MULTIPLIER

        # Generate Signals based on Bollinger Bands and Volume Confirmation
        # Buy signal: Price closes below the lower band AND current volume exceeds the threshold
        if current_close < lower_band and current_volume > volume_confirmation_threshold:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id=RULE_ID
            ))
        # Sell signal: Price closes above the upper band AND current volume exceeds the threshold
        elif current_close > upper_band and current_volume > volume_confirmation_threshold:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id=RULE_ID
            ))

    return signals