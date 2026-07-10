from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Constants ---
BB_PERIOD = 20
BB_STD_DEV = 2.0
# Minimum candles required: BB_PERIOD for the Bollinger Band calculation for Candle A,
# plus one more candle (Candle B) for the re-entry confirmation.
MIN_CANDLES_FOR_RULE = BB_PERIOD + 1

# --- Bollinger Band Calculation ---
def _calculate_bollinger_bands_for_window(
    candles: list[WarmCandle], period: int, std_dev: float
) -> tuple[float, float, float] | tuple[None, None, None]:
    """
    Calculates Bollinger Bands for a specific window of candles.
    This function expects exactly `period` number of candles in the input list.
    Returns (SMA, Upper Band, Lower Band) or (None, None, None) if data is insufficient.
    """
    if len(candles) != period:
        return None, None, None

    closes = np.array([c.close for c in candles])
    
    sma = np.mean(closes)
    std = np.std(closes)
    
    upper_band = sma + std * std_dev
    lower_band = sma - std * std_dev
    
    return sma, upper_band, lower_band


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Bollinger Band Reversal with Price Re-entry Confirmation rule.

    Emits a Buy signal when:
    1. Candle A (previous candle) closes below the Lower Bollinger Band.
    2. Candle B (current candle) closes back above the Lower Bollinger Band.

    Emits a Sell signal when:
    1. Candle A (previous candle) closes above the Upper Bollinger Band.
    2. Candle B (current candle) closes back below the Upper Bollinger Band.
    """
    signals: list[BuySignal | SellSignal] = []
    rule_id = "ae613161-dcd4-4e0f-90db-ec068f76327b" # From idea_id

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough candles for Bollinger Band calculation for Candle A
        # and to have Candle B for confirmation.
        if len(warm_candles) < MIN_CANDLES_FOR_RULE:
            continue

        # Iterate through candles starting from the point where BB can be calculated for Candle A.
        # 'i' will be the index of Candle B.
        # 'i - 1' will be the index of Candle A.
        # The window for BB calculation for Candle A is warm_candles[i - BB_PERIOD : i].
        for i in range(BB_PERIOD, len(warm_candles)):
            # Define Candle A and Candle B
            candle_A = warm_candles[i - 1] # The candle that potentially breaches the band
            candle_B = warm_candles[i]     # The confirmation candle (re-entry)

            # Get the window of `BB_PERIOD` candles ending at Candle A for BB calculation
            # This means the BBs are calculated based on data *up to and including* candle_A.
            window_for_bb = warm_candles[i - BB_PERIOD : i]
            
            sma_val, upper_band_val, lower_band_val = _calculate_bollinger_bands_for_window(
                window_for_bb, BB_PERIOD, BB_STD_DEV
            )

            if sma_val is None: # Should not happen if loop range is correct, but as safeguard
                continue

            # --- Buy Signal Logic ---
            # Condition 1: Candle A closes below the Lower Bollinger Band
            if candle_A.close < lower_band_val:
                # Condition 2: Candle B closes back above the Lower Bollinger Band
                if candle_B.close > lower_band_val:
                    signals.append(BuySignal(
                        pair=pair,
                        timestamp=candle_B.hour, # Signal at the close of Candle B
                        price=candle_B.close,
                        rule_id=rule_id,
                        confidence=None # Not specified in the rule idea
                    ))

            # --- Sell Signal Logic ---
            # Condition 1: Candle A closes above the Upper Bollinger Band
            if candle_A.close > upper_band_val:
                # Condition 2: Candle B closes back below the Upper Bollinger Band
                if candle_B.close < upper_band_val:
                    signals.append(SellSignal(
                        pair=pair,
                        timestamp=candle_B.hour, # Signal at the close of Candle B
                        price=candle_B.close,
                        rule_id=rule_id,
                        confidence=None # Not specified in the rule idea
                    ))
    return signals