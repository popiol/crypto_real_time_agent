from __future__ import annotations

import numpy as np
import uuid
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle

# Rule constants
BOLLINGER_PERIOD = 20
BB_DEV = 2.0  # Standard deviations for Bollinger Bands
MFI_PERIOD = 14
MFI_OVERSOLD = 20
MFI_OVERBOUGHT = 80
VOL_SMA_PERIOD = 20

# Minimum number of warm candles required to calculate all indicators.
# Bollinger Bands and Volume SMA require `PERIOD` candles.
# MFI requires `MFI_PERIOD + 1` candles for a single calculation.
# To calculate both current MFI and previous MFI, we need `(MFI_PERIOD + 1)` candles for the current MFI,
# and `(MFI_PERIOD + 1)` candles for the previous MFI (which means the data slice for previous MFI ends one candle earlier).
# Thus, total candles needed for MFI_prev is `(MFI_PERIOD + 1) + 1 = MFI_PERIOD + 2`.
MIN_CANDLES_REQUIRED = max(BOLLINGER_PERIOD, MFI_PERIOD + 2, VOL_SMA_PERIOD)

RULE_ID = str(uuid.uuid4()) # Unique identifier for this rule

def _sma(data: np.ndarray, period: int) -> float:
    """Calculates the Simple Moving Average."""
    if len(data) < period:
        return np.nan # Not enough data
    return np.mean(data[-period:])

def _stddev(data: np.ndarray, period: int) -> float:
    """Calculates the Standard Deviation."""
    if len(data) < period:
        return np.nan # Not enough data
    return np.std(data[-period:])

def _calculate_mfi_single(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray, period: int) -> float:
    """
    Calculates the Money Flow Index (MFI) for the latest candle in the provided series.
    Requires at least `period + 1` candles for calculation.
    """
    if len(highs) < period + 1:
        return np.nan

    # Ensure we have exactly `period + 1` candles for the calculation window
    highs_slice = highs[- (period + 1):]
    lows_slice = lows[- (period + 1):]
    closes_slice = closes[- (period + 1):]
    volumes_slice = volumes[- (period + 1):]

    typical_price = (highs_slice + lows_slice + closes_slice) / 3
    money_flow = typical_price * volumes_slice

    positive_money_flow_sum = 0.0
    negative_money_flow_sum = 0.0

    # Iterate over the `period` comparisons (from the second candle to the last)
    # This covers `period` data points for money flow
    for i in range(1, period + 1):
        if typical_price[i] > typical_price[i-1]:
            positive_money_flow_sum += money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_money_flow_sum += money_flow[i]

    if negative_money_flow_sum == 0:
        if positive_money_flow_sum == 0:
            return 50.0 # No money flow, MFI is typically 50
        else:
            return 100.0 # Strong positive flow, no negative flow
    else:
        money_ratio = positive_money_flow_sum / negative_money_flow_sum
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candle data for calculations
        if len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue

        # Extract data from warm candles
        warm_candles = pair_data.warm
        
        # Create numpy arrays for indicator calculations
        closes = np.array([c.close for c in warm_candles])
        highs = np.array([c.high for c in warm_candles])
        lows = np.array([c.low for c in warm_candles])
        volumes = np.array([c.volume for c in warm_candles])

        # Get current and previous candle data for direct comparison
        current_candle: WarmCandle = warm_candles[-1]
        prev_candle: WarmCandle = warm_candles[-2] 

        current_close = current_candle.close
        current_high = current_candle.high
        current_low = current_candle.low
        current_volume = current_candle.volume
        prev_close = prev_candle.close # Guaranteed to exist due to MIN_CANDLES_REQUIRED

        # --- Calculate Bollinger Bands ---
        bb_middle = _sma(closes, BOLLINGER_PERIOD)
        bb_std_dev = _stddev(closes, BOLLINGER_PERIOD)

        if np.isnan(bb_middle) or np.isnan(bb_std_dev) or bb_std_dev == 0:
            continue # Not enough data or std_dev is zero, cannot form valid bands

        bb_upper = bb_middle + (bb_std_dev * BB_DEV)
        bb_lower = bb_middle - (bb_std_dev * BB_DEV)

        # --- Calculate Money Flow Index (MFI) ---
        # MFI for the current candle (uses data up to and including current_candle)
        mfi_current = _calculate_mfi_single(highs, lows, closes, volumes, MFI_PERIOD)
        # MFI for the previous candle (uses data up to and including prev_candle)
        mfi_prev = _calculate_mfi_single(highs[:-1], lows[:-1], closes[:-1], volumes[:-1], MFI_PERIOD)

        if np.isnan(mfi_current) or np.isnan(mfi_prev):
            # This case should ideally be prevented by MIN_CANDLES_REQUIRED
            continue 

        # --- Calculate Volume SMA ---
        vol_sma = _sma(volumes, VOL_SMA_PERIOD)
        if np.isnan(vol_sma):
            # This case should ideally be prevented by MIN_CANDLES_REQUIRED
            continue 

        # --- Get Timestamp for Signal ---
        # The signal is generated based on the latest warm candle's closing conditions.
        ts = current_candle.hour
        
        # --- Generate Signals ---
        # Buy Signal Conditions:
        # 1. Price breached lower band (PREV_CLOSE < BB_LOWER OR LOW < BB_LOWER) AND closed back above it (CLOSE > BB_LOWER)
        # 2. MFI oversold (MFI < MFI_OVERSOLD) AND turning up (MFI > PREV_MFI)
        # 3. Current volume exceeds its SMA (VOLUME > VOL_SMA)

        buy_condition_price = (prev_close < bb_lower or current_low < bb_lower) and current_close > bb_lower
        buy_condition_mfi = mfi_current < MFI_OVERSOLD and mfi_current > mfi_prev
        buy_condition_volume = current_volume > vol_sma

        if buy_condition_price and buy_condition_mfi and buy_condition_volume:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=ts,
                    price=current_close, # Signal price is the current candle's close
                    rule_id=RULE_ID,
                )
            )

        # Sell Signal Conditions:
        # 1. Price breached upper band (PREV_CLOSE > BB_UPPER OR HIGH > BB_UPPER) AND closed back below it (CLOSE < BB_UPPER)
        # 2. MFI overbought (MFI > MFI_OVERBOUGHT) AND turning down (MFI < PREV_MFI)
        # 3. Current volume exceeds its SMA (VOLUME > VOL_SMA)

        sell_condition_price = (prev_close > bb_upper or current_high > bb_upper) and current_close < bb_upper
        sell_condition_mfi = mfi_current > MFI_OVERBOUGHT and mfi_current < mfi_prev
        sell_condition_volume = current_volume > vol_sma

        if sell_condition_price and sell_condition_mfi and sell_condition_volume:
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=ts,
                    price=current_close, # Signal price is the current candle's close
                    rule_id=RULE_ID,
                )
            )

    return signals