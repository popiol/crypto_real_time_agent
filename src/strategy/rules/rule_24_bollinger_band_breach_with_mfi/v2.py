"""Rule 7f20005f-8acd-45a2-9bb7-f93ee74c7f99 — Enhance Bollinger Band Breach with MFI by Adding Volume Confirmation."""
from __future__ import annotations
import numpy as np
import talib as ta
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

# Parameters for Bollinger Bands
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0 # Corresponds to BB_STD_DEV in pseudocode

# Parameters for MFI
MFI_PERIOD = 14
MFI_OVERSOLD_THRESHOLD = 20 # Updated from 30 based on pseudocode
MFI_OVERBOUGHT_THRESHOLD = 80 # Updated from 70 based on pseudocode

# Parameters for Volume Confirmation
VOLUME_MA_PERIOD = 20
VOLUME_MULTIPLIER = 1.5 # e.g., current volume > 1.5 * VOLUME_MA

# Minimum number of warm candles required for calculations.
# We need enough candles for all indicators (BB, MFI, Volume SMA)
# to produce a value at the latest candle, and one more for the 'previous' MFI value
# for crossover detection.
MIN_CANDLES_REQUIRED = max(BB_PERIOD, MFI_PERIOD + 1, VOLUME_MA_PERIOD)

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough data for calculations
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract necessary data for TA-Lib.
        # Data is assumed to be ordered from oldest to newest.
        high_prices = np.array([c.high for c in warm_candles], dtype=float)
        low_prices = np.array([c.low for c in warm_candles], dtype=float)
        close_prices = np.array([c.close for c in warm_candles], dtype=float)
        volumes = np.array([c.volume for c in warm_candles], dtype=float)

        # Calculate Bollinger Bands
        # TA-Lib BBANDS returns (upperband, middleband, lowerband)
        upper_band, _, lower_band = ta.BBANDS(
            close_prices,
            timeperiod=BB_PERIOD,
            nbdevup=BB_STD_DEV_MULTIPLIER,
            nbdevdn=BB_STD_DEV_MULTIPLIER,
            matype=0  # 0 for SMA, as implied by pseudocode
        )

        # Calculate Money Flow Index (MFI)
        mfi = ta.MFI(
            high_prices,
            low_prices,
            close_prices,
            volumes,
            timeperiod=MFI_PERIOD
        )

        # Calculate Average Volume
        avg_volume_ma = ta.SMA(
            volumes,
            timeperiod=VOLUME_MA_PERIOD
        )

        # Ensure indicators have valid values at the latest points needed.
        # The last element in the arrays corresponds to the most recent candle.
        # We need the last two MFI values for crossover detection.
        # We also need valid current values for BB and Volume MA.
        if (
            np.isnan(upper_band[-1]) or np.isnan(lower_band[-1]) or
            np.isnan(mfi[-1]) or np.isnan(mfi[-2]) or
            np.isnan(avg_volume_ma[-1])
        ):
            continue

        # Get current and previous values for signal conditions
        current_close = close_prices[-1]
        current_upper_band = upper_band[-1]
        current_lower_band = lower_band[-1]
        current_mfi = mfi[-1]
        prev_mfi = mfi[-2]
        current_volume = volumes[-1]
        current_avg_volume = avg_volume_ma[-1]

        # Volume Confirmation Condition
        high_volume_confirmation = (current_volume > (current_avg_volume * VOLUME_MULTIPLIER))

        # Buy Signal Condition:
        # Price closes below lower Bollinger Band
        # AND MFI crosses above its oversold threshold
        # AND current trading volume is significantly higher than its recent average
        buy_signal_condition = (
            current_close < current_lower_band and
            current_mfi > MFI_OVERSOLD_THRESHOLD and
            prev_mfi <= MFI_OVERSOLD_THRESHOLD and
            high_volume_confirmation
        )

        # Sell Signal Condition:
        # Price closes above upper Bollinger Band
        # AND MFI crosses below its overbought threshold
        # AND current trading volume is significantly higher than its recent average
        sell_signal_condition = (
            current_close > current_upper_band and
            current_mfi < MFI_OVERBOUGHT_THRESHOLD and
            prev_mfi >= MFI_OVERBOUGHT_THRESHOLD and
            high_volume_confirmation
        )

        latest_candle = warm_candles[-1]
        trade_timestamp = latest_candle.hour
        trade_price = latest_candle.close # Use the close price of the latest candle for the signal

        if buy_signal_condition:
            signals.append(BuySignal(
                pair=pair,
                timestamp=trade_timestamp,
                price=trade_price,
                rule_id="7f20005f-8acd-45a2-9bb7-f93ee74c7f99" # Unique ID for this specific rule
            ))
        elif sell_signal_condition:
            signals.append(SellSignal(
                pair=pair,
                timestamp=trade_timestamp,
                price=trade_price,
                rule_id="7f20005f-8acd-45a2-9bb7-f93ee74c7f99" # Unique ID for this specific rule
            ))

    return signals