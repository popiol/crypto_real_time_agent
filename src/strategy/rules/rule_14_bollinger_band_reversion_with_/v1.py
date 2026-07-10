from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

# Rule parameters
PERIOD = 20
STD_DEV_MULTIPLIER = 2

# Minimum number of candles required to calculate Bollinger Bands for the last two candles (N and N-1).
# To calculate BB for candle N, we need 'PERIOD' preceding candles.
# To calculate BB for candle N-1, we need 'PERIOD' preceding candles.
# The BB value at index `k` in the calculated arrays corresponds to the candle at `warm_candles[k + PERIOD - 1]`.
# To get BB for `warm_candles[-1]` (candle N), we need `warm_candles[-1 - (PERIOD - 1)]` up to `warm_candles[-1]`.
# To get BB for `warm_candles[-2]` (candle N-1), we need `warm_candles[-2 - (PERIOD - 1)]` up to `warm_candles[-2]`.
# The earliest candle needed is `warm_candles[-2 - (PERIOD - 1)] = warm_candles[-PERIOD - 1]`.
# So, we need `PERIOD + 1` candles in total (from index 0 to PERIOD).
MIN_CANDLES = PERIOD + 1

def _calculate_bollinger_bands(closes: np.ndarray, period: int, std_dev_multiplier: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates Simple Moving Average (SMA), Upper Bollinger Band, and Lower Bollinger Band.
    The returned arrays will have a length of `len(closes) - period + 1`.
    Each element `k` in the output arrays corresponds to the candle `closes[k + period - 1]`.
    """
    if len(closes) < period:
        return np.array([]), np.array([]), np.array([])

    sma = np.zeros(len(closes) - period + 1)
    std_dev = np.zeros(len(closes) - period + 1)

    for i in range(len(sma)):
        window = closes[i : i + period]
        sma[i] = np.mean(window)
        std_dev[i] = np.std(window)

    upper_band = sma + (std_dev * std_dev_multiplier)
    lower_band = sma - (std_dev * std_dev_multiplier)

    return sma, upper_band, lower_band

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Bollinger Band Reversion with Close-Inside Confirmation trading rule.

    A Buy signal is generated when:
    1. The previous candle closed below the lower band.
    2. The current candle closes back above the lower band (i.e., inside).
    3. The current close is higher than the low of the candle that breached the band (N-1).

    A Sell signal is generated when:
    1. The previous candle closed above the upper band.
    2. The current candle closes back below the upper band (i.e., inside).
    3. The current close is lower than the high of the candle that breached the band (N-1).
    """
    signals: list[BuySignal | SellSignal] = []
    RULE_ID = "7ed16df5-26e8-49d6-bc2b-9bd3b2ea69d6"

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        if len(warm_candles) < MIN_CANDLES:
            continue

        # Extract relevant price data for calculations and comparisons
        closes = np.array([c.close for c in warm_candles])
        lows = np.array([c.low for c in warm_candles])
        highs = np.array([c.high for c in warm_candles])

        # Calculate Bollinger Bands
        _, upper_band, lower_band = _calculate_bollinger_bands(closes, PERIOD, STD_DEV_MULTIPLIER)

        # Determine indices for the current (N) and previous (N-1) candles' Bollinger Band values.
        # The BB arrays (upper_band, lower_band) are indexed such that `bb[k]` corresponds
        # to the candle `warm_candles[k + PERIOD - 1]`.
        # So, for the latest candle `warm_candles[-1]` (N), its BB values are at index:
        # `bb_idx_N = (len(warm_candles) - 1) - (PERIOD - 1) = len(warm_candles) - PERIOD`
        # For the previous candle `warm_candles[-2]` (N-1), its BB values are at index:
        # `bb_idx_N_minus_1 = (len(warm_candles) - 2) - (PERIOD - 1) = len(warm_candles) - PERIOD - 1`
        bb_idx_N = len(warm_candles) - PERIOD
        bb_idx_N_minus_1 = len(warm_candles) - PERIOD - 1

        # Retrieve values for the current candle (N)
        current_candle = warm_candles[-1]
        close_N = current_candle.close
        upper_band_N = upper_band[bb_idx_N]
        lower_band_N = lower_band[bb_idx_N]

        # Retrieve values for the previous candle (N-1)
        previous_candle = warm_candles[-2]
        close_N_minus_1 = previous_candle.close
        high_N_minus_1 = previous_candle.high
        low_N_minus_1 = previous_candle.low
        upper_band_N_minus_1 = upper_band[bb_idx_N_minus_1]
        lower_band_N_minus_1 = lower_band[bb_idx_N_minus_1]

        # Check for Buy Signal
        # Condition 1: Previous candle (N-1) closed below the Lower Band
        # Condition 2: Current candle (N) closes above the Lower Band (i.e., back inside)
        # Condition 3: Current close is higher than the low of candle N-1 (simple bullish confirmation)
        if (close_N_minus_1 < lower_band_N_minus_1 and
            close_N > lower_band_N and
            close_N > low_N_minus_1):
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id=RULE_ID
            ))

        # Check for Sell Signal
        # Condition 1: Previous candle (N-1) closed above the Upper Band
        # Condition 2: Current candle (N) closes below the Upper Band (i.e., back inside)
        # Condition 3: Current close is lower than the high of candle N-1 (simple bearish confirmation)
        if (close_N_minus_1 > upper_band_N_minus_1 and
            close_N < upper_band_N and
            close_N < high_N_minus_1):
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id=RULE_ID
            ))

    return signals