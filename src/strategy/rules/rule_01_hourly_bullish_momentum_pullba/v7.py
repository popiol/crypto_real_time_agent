from __future__ import annotations
from datetime import datetime
import math
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

RULE_ID = "momentum_pullback_v2"

def _calculate_stoch_k(
    closes: list[float], highs: list[float], lows: list[float], period: int, current_idx: int
) -> float:
    """
    Calculates the %K value for a given index within a series,
    using a lookback window of `period` candles ending at `current_idx`.
    """
    # The window starts `period - 1` candles before `current_idx` (inclusive of `current_idx`).
    # For a 14-period, if current_idx is 13, window_start_idx is 0.
    window_start_idx = max(0, current_idx - period + 1)
    
    current_window_lows = lows[window_start_idx : current_idx + 1]
    current_window_highs = highs[window_start_idx : current_idx + 1]
    current_close = closes[current_idx]

    # This defensive check should ideally not be hit if `signal` function ensures enough data
    if not current_window_lows or not current_window_highs:
        return 50.0 # Neutral value if data is unexpectedly missing

    lowest_low = min(current_window_lows)
    highest_high = max(current_window_highs)

    # Avoid division by zero if range is zero, return a neutral value
    if (highest_high - lowest_low) == 0:
        return 50.0
    
    stoch_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
    return stoch_k

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # --- Initial Data Checks ---
        # 1. Need at least 20 warm candles as per pseudocode.
        #    This ensures `prev_candle` exists (index -2) and provides enough history
        #    for a 14-period Stochastic %K and 3-period %D calculation.
        #    (14-period K needs 14 candles, first K at index 13.
        #    If len(warm) is 20, we get K values for indices 13-19, which is 7 values.
        #    This is sufficient for a 3-period D line.)
        # 2. Need at least one hot tick for current price and timestamp.
        
        MIN_WARM_CANDLES = 20
        
        if len(pair_data.warm) < MIN_WARM_CANDLES or not pair_data.hot:
            continue
        
        prev_candle = pair_data.warm[-2]
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_timestamp = current_tick.polled_at

        # --- Condition 1: Strong Bullish Candle (previous complete hourly candle) ---
        # The candle body must be substantial and positive.
        is_strong_bullish = False
        if prev_candle.close > prev_candle.open_price:
            candle_body = prev_candle.close - prev_candle.open_price
            candle_range = prev_candle.high - prev_candle.low
            
            # Ensure candle_range is positive to avoid division by zero and ensure a meaningful candle.
            # Adding a small epsilon as per pseudocode to guard against exact zero.
            if candle_range > 0.000001:
                candle_body_ratio = candle_body / candle_range
                # Check for strong body ratio and significant candle size relative to its open price
                if candle_body_ratio > 0.65 and (candle_body / prev_candle.open_price) > 0.005:
                    is_strong_bullish = True
        
        if not is_strong_bullish:
            continue

        # --- Condition 2: Pullback Range Check ---
        # Current price must be within a defined pullback range.
        # pullback_lower_bound = prev_candle.open - (prev_candle.close - prev_candle.open) * 0.15;
        # pullback_upper_bound = prev_candle.close;
        
        pullback_lower_bound = prev_candle.open_price - (prev_candle.close - prev_candle.open_price) * 0.15
        pullback_upper_bound = prev_candle.close

        is_valid_pullback = (current_price > pullback_lower_bound) and (current_price < pullback_upper_bound)
        
        if not is_valid_pullback:
            continue

        # --- Condition 3: Stochastic Oscillator (relaxed oversold: %K and %D below 60, %K > %D) ---
        stoch_period = 14
        stoch_d_sma_period = 3

        closes = [c.close for c in pair_data.warm]
        highs = [c.high for c in pair_data.warm]
        lows = [c.low for c in pair_data.warm]
        
        all_stoch_k_values = []
        # Calculate %K for each candle starting from when enough data is available for a full period.
        # For a 14-period K, we need to start at index 13 (0-indexed).
        for i in range(len(pair_data.warm)):
            if i >= stoch_period - 1:
                k = _calculate_stoch_k(closes, highs, lows, stoch_period, i)
                all_stoch_k_values.append(k)

        # Ensure we have enough %K values to calculate the %D line (3-period SMA of %K).
        if len(all_stoch_k_values) < stoch_d_sma_period:
            continue 
        
        stoch_k = all_stoch_k_values[-1] # The most recent %K value
        stoch_d = np.mean(all_stoch_k_values[-stoch_d_sma_period:]) # 3-period SMA of %K

        # Relaxed oversold condition: both %K and %D below 60, and %K is above %D (or crossing up).
        if not (stoch_k < 60 and stoch_d < 60 and stoch_k > stoch_d):
            continue

        # All conditions met, generate Buy Signal
        signals.append(BuySignal(
            pair=pair,
            timestamp=current_timestamp,
            price=current_price,
            rule_id=RULE_ID,
            confidence=1.0,
        ))

    return signals