from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Parameters
PERIOD = 20
STD_DEV_MULTIPLIER = 2.2
VOLUME_MA_PERIOD = 30
VOLUME_SPIKE_MULTIPLIER = 1.5

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on Bollinger Bands with Volume Spike Confirmation.

    A Buy signal is generated when the price closes below the lower Bollinger Band
    and there's a simultaneous volume surge (e.g., above a moving average of volume).
    A Sell signal is generated when the price closes above the upper Bollinger Band
    with a similar volume spike.
    """
    signals: list[BuySignal | SellSignal] = []

    min_required_candles = max(PERIOD, VOLUME_MA_PERIOD)

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # Ensure enough candles are available for calculations
        if len(candles) < min_required_candles:
            continue

        # Extract close prices and volumes from the candles
        # We need the most recent candles for calculations
        close_prices = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Calculate Bollinger Bands using the last 'PERIOD' candles
        bb_close_prices = close_prices[-PERIOD:]
        sma = np.mean(bb_close_prices)
        std_dev = np.std(bb_close_prices)
        upper_band = sma + (std_dev * STD_DEV_MULTIPLIER)
        lower_band = sma - (std_dev * STD_DEV_MULTIPLIER)

        # Calculate Volume Moving Average using the last 'VOLUME_MA_PERIOD' candles
        volume_ma_data = volumes[-VOLUME_MA_PERIOD:]
        volume_ma = np.mean(volume_ma_data)

        # Get the latest candle's data for signal generation
        latest_candle: WarmCandle = candles[-1]
        current_close = latest_candle.close
        current_volume = latest_candle.volume
        current_timestamp = latest_candle.hour

        # Generate Signals
        volume_spike_threshold = volume_ma * VOLUME_SPIKE_MULTIPLIER

        # Buy signal: Close below lower band AND volume spike
        if current_close < lower_band and current_volume > volume_spike_threshold:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id="c45c3d77-27f0-439e-9ff8-d7117cbc2bbf"
            ))
        # Sell signal: Close above upper band AND volume spike
        elif current_close > upper_band and current_volume > volume_spike_threshold:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id="c45c3d77-27f0-439e-9ff8-d7117cbc2bbf"
            ))

    return signals