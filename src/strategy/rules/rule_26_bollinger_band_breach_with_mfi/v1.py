from __future__ import annotations
import numpy as np
import statistics # Not explicitly used for SMA/StdDev due to numpy, but good practice to import if needed.
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle
from datetime import datetime

# --- Parameters ---
PERIOD_BB = 20
STD_DEV_BB = 2.0
PERIOD_MFI = 14
OVERSOLD_MFI_THRESHOLD = 20
OVERBOUGHT_MFI_THRESHOLD = 80
VOLUME_AVG_PERIOD = 20
VOLUME_MULTIPLIER = 1.5

# Minimum number of candles required for all calculations
# Max of (PERIOD_BB, PERIOD_MFI + 1 for MFI calculation, VOLUME_AVG_PERIOD)
MIN_CANDLES_REQUIRED = max(PERIOD_BB, PERIOD_MFI + 1, VOLUME_AVG_PERIOD)

# Rule ID as specified in the idea
RULE_ID = "b5b883ef-a44d-4278-bfad-90f3df0ea4eb"

# --- Helper Functions ---

def calculate_mfi(high_arr: np.ndarray, low_arr: np.ndarray, close_arr: np.ndarray, volume_arr: np.ndarray, period: int) -> float:
    """
    Calculates the Money Flow Index (MFI) for the latest data point given historical arrays.
    Requires `period + 1` data points to calculate the MFI for the last candle.
    """
    if len(high_arr) < period + 1:
        return np.nan

    # Slice the data to the required length for the latest MFI calculation
    # We need (period + 1) values to calculate 'period' differences and then sum over 'period'
    high_sliced = high_arr[-(period + 1):]
    low_sliced = low_arr[-(period + 1):]
    close_sliced = close_arr[-(period + 1):]
    volume_sliced = volume_arr[-(period + 1):]

    tp = (high_sliced + low_sliced + close_sliced) / 3
    money_flow = tp * volume_sliced

    positive_mf = np.zeros_like(money_flow)
    negative_mf = np.zeros_like(money_flow)

    # Calculate positive and negative money flow based on TP changes
    for i in range(1, len(tp)):
        if tp[i] > tp[i-1]:
            positive_mf[i] = money_flow[i]
        elif tp[i] < tp[i-1]:
            negative_mf[i] = money_flow[i]

    # Sum over the MFI period. We need the sum of the last 'period' values.
    pmf_sum = np.sum(positive_mf[-period:])
    nmf_sum = np.sum(negative_mf[-period:])

    if nmf_sum == 0:
        money_ratio = np.inf # Avoid division by zero; MFI will be 100
    else:
        money_ratio = pmf_sum / nmf_sum

    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on Bollinger Band breach, MFI, and adaptive volume confirmation.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough data for all indicator calculations
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract data into numpy arrays for efficient calculations
        close_prices = np.array([c.close for c in warm_candles])
        high_prices = np.array([c.high for c in warm_candles])
        low_prices = np.array([c.low for c in warm_candles])
        volumes = np.array([c.volume for c in warm_candles])

        current_candle = warm_candles[-1]
        current_price = current_candle.close
        current_volume = current_candle.volume
        timestamp = current_candle.hour # Use candle hour as timestamp for the signal

        # --- Calculate Bollinger Bands ---
        bb_close_window = close_prices[-PERIOD_BB:]
        sma_bb = np.mean(bb_close_window)
        std_dev_bb = np.std(bb_close_window)

        upper_bb = sma_bb + (STD_DEV_BB * std_dev_bb)
        lower_bb = sma_bb - (STD_DEV_BB * std_dev_bb)

        # --- Calculate Money Flow Index (MFI) ---
        mfi = calculate_mfi(high_prices, low_prices, close_prices, volumes, PERIOD_MFI)
        if np.isnan(mfi):
            # This should ideally be caught by MIN_CANDLES_REQUIRED, but acts as a safeguard
            continue

        # --- Calculate Adaptive Volume ---
        volume_avg_window = volumes[-VOLUME_AVG_PERIOD:]
        avg_volume = np.mean(volume_avg_window)
        is_high_volume = current_volume > (avg_volume * VOLUME_MULTIPLIER)

        # --- Generate Buy Signal ---
        if (current_price < lower_bb and
                mfi < OVERSOLD_MFI_THRESHOLD and
                is_high_volume):
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID
            ))

        # --- Generate Sell Signal ---
        elif (current_price > upper_bb and
              mfi > OVERBOUGHT_MFI_THRESHOLD and
              is_high_volume):
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID
            ))

    return signals