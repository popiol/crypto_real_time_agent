from __future__ import annotations
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

RULE_ID = "relaxed_stoch_pullback_v1"

def _calculate_stoch_k(
    closes: list[float], highs: list[float], lows: list[float], period: int, current_idx: int
) -> float:
    """
    Calculates the %K value for a given index within a series,
    using a lookback window that adapts to available data if less than `period`.
    """
    # Determine the actual lookback window for HH/LL
    # The window ends at current_idx and goes back `period` candles, or to the beginning of the list if shorter.
    window_start_idx = max(0, current_idx - period + 1)
    
    current_window_lows = lows[window_start_idx : current_idx + 1]
    current_window_highs = highs[window_start_idx : current_idx + 1]
    current_close = closes[current_idx]

    # This defensive check should ideally not be hit if current_idx is valid
    if not current_window_lows or not current_window_highs:
        return 0.0

    lowest_low = min(current_window_lows)
    highest_high = max(current_window_highs)

    # Avoid division by zero if range is zero
    if (highest_high - lowest_low) == 0:
        return 50.0  # Neutral value by convention
    
    stoch_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
    return stoch_k

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Initial Data Check:
        # Stochastic Oscillator lookback is 14 periods.
        # We also need `prev_candle` (data.warm[-2]), which requires at least 2 warm candles.
        # The pseudocode's primary data check is `LENGTH(data.warm) < 14`.
        # We also need at least one hot tick for `current_price`.
        if len(pair_data.warm) < 14 or not pair_data.hot:
            continue
        
        # Ensure we have at least two candles to access data.warm[-2]
        if len(pair_data.warm) < 2:
            continue

        # Define the previous completed hourly candle for momentum identification
        # Pseudocode specifies `data.warm[-2]` for the strong bullish candle.
        prev_candle = pair_data.warm[-2]
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_timestamp = current_tick.polled_at

        # Condition 1: Identify a Strong Bullish Momentum Candle (previous hourly candle)
        # Pseudocode: prev_candle.close > prev_candle.open AND candle_body_ratio > 0.6
        
        is_bullish_momentum = False
        if prev_candle.close > prev_candle.open_price: # Candle is bullish
            candle_body_size = prev_candle.close - prev_candle.open_price
            candle_range = prev_candle.high - prev_candle.low

            # If candle is bullish, candle_range must be > 0 (unless it's a zero-range candle where close=open=high=low, which contradicts close > open).
            if candle_range > 0: 
                candle_body_ratio = candle_body_size / candle_range
                if candle_body_ratio > 0.6:
                    is_bullish_momentum = True
            # If candle_range is 0, it means high=low=open=close, which contradicts prev_candle.close > prev_candle.open_price.
            # Therefore, candle_range > 0 is implied for a bullish candle.
        
        if not is_bullish_momentum:
            continue

        # Condition 2: Check for a Shallow Pullback from the Strong Bullish Candle's Close
        # Current price must be below the previous candle's close but still above its open.
        # Pseudocode: current_price < prev_candle.close AND current_price > prev_candle.open
        if not (current_price < prev_candle.close and current_price > prev_candle.open_price):
            continue

        # Pullback depth must be positive (actual pullback) and less than 60% of the strong candle's body size
        # Pseudocode: pullback_depth > 0 AND (pullback_depth / strong_candle_body_size) < 0.6
        pullback_depth = prev_candle.close - current_price

        # `candle_body_size` was calculated in Condition 1. It must be positive for a strong bullish candle.
        # Defensive check, though `is_bullish_momentum` should ensure `candle_body_size > 0`.
        if candle_body_size <= 0:
            continue 

        if not (pullback_depth > 0 and (pullback_depth / candle_body_size) < 0.6):
            continue

        # Condition 3: Relaxed Oversold Stochastic Oscillator Confirmation
        stoch_period = 14
        stoch_d_sma_period = 3 # D is typically a 3-period Simple Moving Average of K

        closes = [c.close for c in pair_data.warm]
        highs = [c.high for c in pair_data.warm]
        lows = [c.low for c in pair_data.warm]
        
        all_stoch_k_values = []
        # Calculate %K for each candle in `data.warm`. The `_calculate_stoch_k` helper
        # adapts the lookback window to the available data for earlier candles if needed.
        for i in range(len(pair_data.warm)):
            k = _calculate_stoch_k(closes, highs, lows, stoch_period, i)
            all_stoch_k_values.append(k)

        # We need at least `stoch_d_sma_period` K values to calculate D.
        # Since `len(pair_data.warm)` is guaranteed to be at least 14 by the initial check,
        # `len(all_stoch_k_values)` will also be at least 14, so `[-stoch_d_sma_period:]` is safe.
        stoch_k = all_stoch_k_values[-1] # The most recent %K value
        stoch_d = sum(all_stoch_k_values[-stoch_d_sma_period:]) / stoch_d_sma_period # 3-period SMA of %K

        # Check for Stochastic K and D below 40, with K crossing above D or remaining above D in the oversold zone
        # Pseudocode: (stoch_k < 40 AND stoch_d < 40 AND stoch_k > stoch_d)
        if (stoch_k < 40 and stoch_d < 40 and stoch_k > stoch_d):
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))

    return signals