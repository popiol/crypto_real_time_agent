from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal

# --- Parameters for Bollinger Bands ---
BB_PERIOD = 20  # Period for Simple Moving Average (SMA)
BASE_STD_DEV_MULTIPLIER = 2.0  # Default multiplier for average volatility
VOLATILITY_PERIOD = 20  # Period for calculating current BB std dev
LONG_TERM_VOLATILITY_LOOKBACK = 60  # Period to assess market's general volatility regime
VOLATILITY_SENSITIVITY = 0.5  # How much the multiplier changes per unit of volatility deviation

# Optional: Clamp ADAPTIVE_MULTIPLIER to a sensible range
MIN_MULTIPLIER = 1.5
MAX_MULTIPLIER = 3.0

# Minimum number of warm candles required for calculations.
# To calculate AVERAGE_BB_STD_DEV for the current point, we need:
# 1. VOLATILITY_PERIOD candles to get the current BB_STD_DEV.
# 2. LONG_TERM_VOLATILITY_LOOKBACK previous BB_STD_DEV values.
# This means the earliest candle needed is `VOLATILITY_PERIOD + LONG_TERM_VOLATILITY_LOOKBACK - 1` periods ago.
MIN_CANDLES = VOLATILITY_PERIOD + LONG_TERM_VOLATILITY_LOOKBACK - 1
# Ensure BB_PERIOD is also covered, though typically it will be smaller than MIN_CANDLES
MIN_CANDLES = max(MIN_CANDLES, BB_PERIOD)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for all lookback periods
        if len(pair_data.warm) < MIN_CANDLES:
            continue
        # Ensure we have recent tick data for current price and timestamp
        if not pair_data.hot:
            continue

        # Convert warm candle closes to a numpy array for efficient calculations
        closes = np.array([c.close for c in pair_data.warm])

        # --- Calculate Simple Moving Average (SMA) for the band center ---
        # SMA is based on BB_PERIOD, using the most recent candles
        sma = np.mean(closes[-BB_PERIOD:])

        # --- Calculate the standard deviation for the Bollinger Bands (current volatility) ---
        # This is based on VOLATILITY_PERIOD, using the most recent candles
        bb_std_dev_current = np.std(closes[-VOLATILITY_PERIOD:])

        # If current std dev is zero (e.g., flat price for the period), bands collapse to SMA.
        # In such a scenario, no price can cross the bands, so we skip signal generation.
        if bb_std_dev_current == 0:
            continue

        # --- Calculate a longer-term average of the standard deviation to determine the volatility 'baseline' ---
        # We need a series of BB_STD_DEV values over LONG_TERM_VOLATILITY_LOOKBACK periods.
        # Iterate backwards from the most recent point (i=0) up to LONG_TERM_VOLATILITY_LOOKBACK periods ago.
        std_dev_history = []
        num_closes = len(closes)
        for i in range(LONG_TERM_VOLATILITY_LOOKBACK):
            # The window for the i-th previous standard deviation ends at `num_closes - i`.
            # The window starts at `num_closes - i - VOLATILITY_PERIOD`.
            window_start = num_closes - i - VOLATILITY_PERIOD
            window_end = num_closes - i
            
            # This check is technically redundant due to MIN_CANDLES, but adds robustness.
            if window_start < 0:
                break 
            
            std_dev_history.append(np.std(closes[window_start:window_end]))
        
        # Calculate the average of these historical standard deviations.
        # If std_dev_history is empty (shouldn't happen with correct MIN_CANDLES), fallback.
        if not std_dev_history:
            average_bb_std_dev = 0
        else:
            average_bb_std_dev = np.mean(std_dev_history)

        # --- Determine the adaptive multiplier ---
        adaptive_multiplier = BASE_STD_DEV_MULTIPLIER
        if average_bb_std_dev > 0:
            volatility_ratio = bb_std_dev_current / average_bb_std_dev
            adaptive_multiplier = BASE_STD_DEV_MULTIPLIER * (1 + (volatility_ratio - 1) * VOLATILITY_SENSITIVITY)
        # else: adaptive_multiplier remains BASE_STD_DEV_MULTIPLIER as initialized

        # Clamp ADAPTIVE_MULTIPLIER to a sensible range
        adaptive_multiplier = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, adaptive_multiplier))

        # --- Calculate Upper and Lower Bollinger Bands ---
        upper_band = sma + (adaptive_multiplier * bb_std_dev_current)
        lower_band = sma - (adaptive_multiplier * bb_std_dev_current)

        # --- Generate signals ---
        # The current price is the last_price from the most recent tick in hot data
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        if current_price < lower_band:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif current_price > upper_band:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals