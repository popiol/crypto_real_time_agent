from __future__ import annotations
import numpy as np
import talib as ta
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

# Parameters
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0
MFI_PERIOD = 14
MFI_OVERSOLD_THRESHOLD = 30
MFI_OVERBOUGHT_THRESHOLD = 70

# Minimum number of warm candles required for calculations.
# We need at least max(BB_PERIOD, MFI_PERIOD) candles for the indicators
# to produce a value, and one more for the 'previous' MFI value.
MIN_CANDLES_REQUIRED = max(BB_PERIOD, MFI_PERIOD) + 1

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

        # Ensure indicators have valid values at the latest points needed.
        # The last element in the arrays corresponds to the most recent candle.
        # We need the last two MFI values for crossover detection.
        if (
            np.isnan(upper_band[-1]) or np.isnan(lower_band[-1]) or
            np.isnan(mfi[-1]) or np.isnan(mfi[-2])
        ):
            continue

        # Get current and previous values for signal conditions
        current_close = close_prices[-1]
        current_upper_band = upper_band[-1]
        current_lower_band = lower_band[-1]
        current_mfi = mfi[-1]
        prev_mfi = mfi[-2]

        # Buy Signal Condition:
        # Price closes below lower Bollinger Band AND MFI crosses above its oversold threshold
        buy_signal_condition = (
            current_close < current_lower_band and
            current_mfi > MFI_OVERSOLD_THRESHOLD and
            prev_mfi <= MFI_OVERSOLD_THRESHOLD
        )

        # Sell Signal Condition:
        # Price closes above upper Bollinger Band AND MFI crosses below its overbought threshold
        sell_signal_condition = (
            current_close > current_upper_band and
            current_mfi < MFI_OVERBOUGHT_THRESHOLD and
            prev_mfi >= MFI_OVERBOUGHT_THRESHOLD
        )

        latest_candle = warm_candles[-1]
        trade_timestamp = latest_candle.hour
        trade_price = latest_candle.close # Use the close price of the latest candle for the signal

        if buy_signal_condition:
            signals.append(BuySignal(
                pair=pair,
                timestamp=trade_timestamp,
                price=trade_price,
                rule_id="1d7cf5f3-28a3-4c5e-a4c5-c388190e660d"
            ))
        elif sell_signal_condition:
            signals.append(SellSignal(
                pair=pair,
                timestamp=trade_timestamp,
                price=trade_price,
                rule_id="1d7cf5f3-28a3-4c5e-a4c5-c388190e660d"
            ))

    return signals