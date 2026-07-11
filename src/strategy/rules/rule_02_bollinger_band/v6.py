from __future__ import annotations

import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal

# Rule parameters
WINDOW = 20  # Number of periods for SMA and STDDEV calculation
NUM_STD_DEV = 2.0  # Standard deviations for the primary Bollinger Bands
MIN_DURATION = 2  # Number of consecutive periods price must be beyond the band
DEEP_EXCURSION_FACTOR = 0.5  # Additional std dev for "deep" excursion (e.g., 2.0 + 0.5 = 2.5 std dev)

# Minimum number of warm candles required to perform calculations
MIN_CANDLES_FOR_CALC = WINDOW


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for Bollinger Band calculations
        if len(pair_data.warm) < MIN_CANDLES_FOR_CALC:
            continue

        # Extract close prices from warm candles
        all_closes = np.array([c.close for c in pair_data.warm])

        # Calculate Bollinger Bands based on the most recent 'WINDOW' candles
        # These bands are static for the current evaluation
        closes_for_bb_calc = all_closes[-WINDOW:]
        
        mid_band = np.mean(closes_for_bb_calc)
        std_dev = np.std(closes_for_bb_calc)

        if std_dev == 0:
            # Avoid division by zero or meaningless bands if standard deviation is zero
            continue

        upper_band = mid_band + (NUM_STD_DEV * std_dev)
        lower_band = mid_band - (NUM_STD_DEV * std_dev)

        # Calculate the "deep" excursion bands
        deep_upper_band = mid_band + ((NUM_STD_DEV + DEEP_EXCURSION_FACTOR) * std_dev)
        deep_lower_band = mid_band - ((NUM_STD_DEV + DEEP_EXCURSION_FACTOR) * std_dev)

        current_close_price = all_closes[-1]  # The close price of the most recent candle
        # Use the timestamp of the most recent warm candle for the signal
        timestamp = pair_data.warm[-1].hour 

        # --- Buy Signal Logic ---
        buy_triggered = False

        # Check for deep excursion below the lower band
        if current_close_price < deep_lower_band:
            buy_triggered = True

        # If not triggered by deep excursion, check for duration below the lower band
        if not buy_triggered:
            consecutive_below_lower = 0
            # Iterate backwards from the current candle for MIN_DURATION periods
            # all_closes[-1] is the current candle, all_closes[-2] is the previous, etc.
            for i in range(MIN_DURATION):
                # Ensure we don't go out of bounds of available warm candles
                if len(all_closes) < (i + 1):
                    break  # Not enough historical data for the duration check
                
                historical_close = all_closes[-(i + 1)] # Get the close price for the i-th past candle (0-indexed from current)
                if historical_close < lower_band:
                    consecutive_below_lower += 1
                else:
                    break  # Streak broken

            if consecutive_below_lower >= MIN_DURATION:
                buy_triggered = True

        if buy_triggered:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_close_price,
            ))

        # --- Sell Signal Logic ---
        sell_triggered = False

        # Check for deep excursion above the upper band
        if current_close_price > deep_upper_band:
            sell_triggered = True

        # If not triggered by deep excursion, check for duration above the upper band
        if not sell_triggered:
            consecutive_above_upper = 0
            for i in range(MIN_DURATION):
                if len(all_closes) < (i + 1):
                    break  # Not enough historical data for the duration check
                
                historical_close = all_closes[-(i + 1)]
                if historical_close > upper_band:
                    consecutive_above_upper += 1
                else:
                    break  # Streak broken

            if consecutive_above_upper >= MIN_DURATION:
                sell_triggered = True

        if sell_triggered:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_close_price,
            ))

    return signals