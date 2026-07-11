from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Constants ---
BB_PERIOD = 20  # Period for Bollinger Bands SMA and StdDev
BB_DEV = 2.0    # Number of standard deviations for Bollinger Bands
VOLUME_AVG_PERIOD = 20 # Period for calculating average volume
VOLUME_MULTIPLIER = 1.5 # Multiplier for current volume confirmation to be considered 'significant'

# Minimum number of candles required for the rule:
# BB_PERIOD candles for the SMA/StdDev calculation (e.g., c_0 to c_19)
# + 1 for the 'previous_candle' (e.g., c_20)
# + 1 for the 'current_candle' (e.g., c_21)
# Total = BB_PERIOD + 2 candles.
# Example: If BB_PERIOD=20, we need 22 candles.
MIN_CANDLES_FOR_RULE = BB_PERIOD + 2

# --- Helper Functions ---

def _is_bullish_engulfing(prev_candle: WarmCandle, curr_candle: WarmCandle) -> bool:
    """
    Checks for a bullish engulfing pattern.
    Requires:
    1. Previous candle is bearish.
    2. Current candle is bullish.
    3. Current candle's body completely engulfs the previous candle's body.
    """
    # 1. Previous candle must be bearish (close < open)
    if prev_candle.close >= prev_candle.open_price:
        return False

    # 2. Current candle must be bullish (close > open)
    if curr_candle.close <= curr_candle.open_price:
        return False

    # 3. Current candle's body must engulf the previous candle's body
    # This means the current open is below the previous close,
    # AND the current close is above the previous open.
    return curr_candle.open_price < prev_candle.close and \
           curr_candle.close > prev_candle.open_price

def _is_bearish_engulfing(prev_candle: WarmCandle, curr_candle: WarmCandle) -> bool:
    """
    Checks for a bearish engulfing pattern.
    Requires:
    1. Previous candle is bullish.
    2. Current candle is bearish.
    3. Current candle's body completely engulfs the previous candle's body.
    """
    # 1. Previous candle must be bullish (close > open)
    if prev_candle.close <= prev_candle.open_price:
        return False

    # 2. Current candle must be bearish (close < open)
    if curr_candle.close >= curr_candle.open_price:
        return False

    # 3. Current candle's body must engulf the previous candle's body
    # This means the current open is above the previous close,
    # AND the current close is below the previous open.
    return curr_candle.open_price > prev_candle.close and \
           curr_candle.close < prev_candle.open_price

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the "Bollinger Band Breach with Engulfing Candlestick Reversal and Volume Confirmation" rule.
    Identifies high-conviction mean-reversion opportunities when the price breaches the
    Bollinger Bands, immediately followed by an engulfing candlestick pattern, and confirmed
    by significant trading volume.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # Ensure enough candles are available for BB calculation, previous, and current candle.
        if len(candles) < MIN_CANDLES_FOR_RULE:
            continue

        # --- Bollinger Band Calculation ---
        # The Bollinger Bands are calculated based on data *prior to* the current candle.
        # Specifically, we need BB_PERIOD candles ending with the 'previous_candle' (candles[-2]).
        # So, we slice from `-(BB_PERIOD + 1)` up to `-1` to get these BB_PERIOD candles.
        bb_calculation_candles = candles[-(BB_PERIOD + 1):-1]

        # Double-check that the slice resulted in enough candles for BB_PERIOD
        if len(bb_calculation_candles) < BB_PERIOD:
            continue
        
        close_prices_for_bb = [c.close for c in bb_calculation_candles]
        
        # Calculate Simple Moving Average (SMA) and Standard Deviation (StdDev)
        sma = statistics.mean(close_prices_for_bb)
        std_dev = statistics.stdev(close_prices_for_bb)
        
        # Calculate Upper and Lower Bollinger Bands
        upper_band = sma + (BB_DEV * std_dev)
        lower_band = sma - (BB_DEV * std_dev)

        # --- Candlestick Pattern Detection ---
        previous_candle = candles[-2]
        current_candle = candles[-1]

        # --- Volume Confirmation ---
        # Calculate average volume over VOLUME_AVG_PERIOD candles,
        # also ending with the 'previous_candle'.
        volume_calculation_candles = candles[-(VOLUME_AVG_PERIOD + 1):-1]
        
        if len(volume_calculation_candles) < VOLUME_AVG_PERIOD:
            continue
        
        avg_volume = statistics.mean([c.volume for c in volume_calculation_candles])
        volume_threshold = avg_volume * VOLUME_MULTIPLIER

        # --- Check for Buy Signal ---
        # Conditions:
        # 1. Current candle's close price breaches below the Lower Bollinger Band.
        # 2. A bullish engulfing pattern is formed between the previous and current candle.
        # 3. Current candle's volume is significantly higher than the average volume.
        if (current_candle.close < lower_band and
            _is_bullish_engulfing(previous_candle, current_candle) and
            current_candle.volume > volume_threshold):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id="616f2f8f-51c1-49c9-be93-3e8900e80921",
                confidence=0.9 # High conviction signal
            ))

        # --- Check for Sell Signal ---
        # Conditions:
        # 1. Current candle's close price breaches above the Upper Bollinger Band.
        # 2. A bearish engulfing pattern is formed between the previous and current candle.
        # 3. Current candle's volume is significantly higher than the average volume.
        elif (current_candle.close > upper_band and
              _is_bearish_engulfing(previous_candle, current_candle) and
              current_candle.volume > volume_threshold):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id="616f2f8f-51c1-49c9-be93-3e8900e80921",
                confidence=0.9 # High conviction signal
            ))

    return signals