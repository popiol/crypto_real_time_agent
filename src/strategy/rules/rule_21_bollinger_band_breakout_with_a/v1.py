from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Rule Parameters
PERIOD_BB = 20
STD_DEV_BB = 2.0
PERIOD_VOL_SMA = 20
VOLUME_MULTIPLIER = 1.5

# Minimum number of warm candles required to perform calculations
MIN_CANDLES_REQUIRED = max(PERIOD_BB, PERIOD_VOL_SMA)

# Rule ID for signals
RULE_ID = "383f0ff1-580b-4add-9e74-24356594f4f1"

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates mean-reversion signals when the price breaks outside the standard Bollinger Bands,
    confirmed by an adaptive surge in trading volume.

    A Buy signal is issued when the price closes below the lower Bollinger Band AND the current
    trading volume is significantly higher (e.g., 1.5 times) than its recent average volume.
    Conversely, a Sell signal is issued when the price closes above the upper Bollinger Band AND
    the current trading volume meets the same adaptive high-volume criterion.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles: list[WarmCandle] = pair_data.warm

        # Ensure we have enough historical candle data for calculations
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract close prices and volumes from the warm candles
        # Using numpy arrays for efficient calculations
        close_prices = np.array([c.close for c in warm_candles])
        volumes = np.array([c.volume for c in warm_candles])

        # Get the most recent candle for current price and volume
        current_candle = warm_candles[-1]
        current_price = current_candle.close
        current_volume = current_candle.volume
        signal_timestamp = current_candle.hour # Use the candle's hour as the signal timestamp

        # Calculate Bollinger Bands for the last PERIOD_BB candles
        # The window for SMA and STDDEV should end with the current candle
        price_window = close_prices[-PERIOD_BB:]
        sma_price = np.mean(price_window)
        std_price = np.std(price_window)

        upper_bb = sma_price + (STD_DEV_BB * std_price)
        lower_bb = sma_price - (STD_DEV_BB * std_price)

        # Calculate Adaptive Volume Threshold for the last PERIOD_VOL_SMA candles
        # The window for SMA should end with the current candle
        volume_window = volumes[-PERIOD_VOL_SMA:]
        sma_volume = np.mean(volume_window)
        volume_threshold = sma_volume * VOLUME_MULTIPLIER

        # Check for Buy Signal
        # Price closes below lower BB AND current volume is above threshold
        if current_price < lower_bb and current_volume > volume_threshold:
            signals.append(BuySignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=current_price,
                rule_id=RULE_ID,
                confidence=None # Confidence not specified in rule, leaving as None
            ))

        # Check for Sell Signal
        # Price closes above upper BB AND current volume is above threshold
        elif current_price > upper_bb and current_volume > volume_threshold:
            signals.append(SellSignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=current_price,
                rule_id=RULE_ID,
                confidence=None # Confidence not specified in rule, leaving as None
            ))

    return signals