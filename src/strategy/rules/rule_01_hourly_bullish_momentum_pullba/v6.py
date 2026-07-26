from __future__ import annotations
from datetime import datetime
import math
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

RULE_ID = "relaxed_bullish_pullback_v2"

def _calculate_stoch_k(
    closes: list[float], highs: list[float], lows: list[float], period: int, current_idx: int
) -> float:
    """
    Calculates the %K value for a given index within a series,
    using a lookback window that adapts to available data if less than `period`.
    """
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
        # --- Initial Data Checks ---
        # 1. Need at least 2 warm candles to identify prev_candle (data.warm[-2]).
        # 2. Need at least 15 warm candles for Stochastic (14-period K, 3-period D smoothing means
        #    a minimum of 14 candles for %K calculation, and 3 K-values for %D smoothing.
        #    The pseudocode specifies 'if len(data.warm) < 15', so we adhere to that as the minimum
        #    for the last K and D values to be calculated, given _calculate_stoch_k's adaptive window).
        # 3. Need at least one hot tick for current price, timestamp, spread.
        # 4. Need at least 2 hot ticks for volatility calculation (to get one return).
        
        MIN_WARM_CANDLES_FOR_STOCH = 15 
        MIN_HOT_TICKS_FOR_VOL = 2 

        if (len(pair_data.warm) < MIN_WARM_CANDLES_FOR_STOCH or 
            not pair_data.hot or 
            len(pair_data.hot) < MIN_HOT_TICKS_FOR_VOL):
            continue
        
        prev_candle = pair_data.warm[-2]
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_timestamp = current_tick.polled_at

        # --- Condition 1: Strong Bullish Candle (previous complete hourly candle) ---
        # The candle body must be substantial and positive.
        is_bullish_momentum = False
        if prev_candle.close > prev_candle.open_price:
            candle_body = prev_candle.close - prev_candle.open_price
            candle_range = prev_candle.high - prev_candle.low
            
            # Ensure candle_range is positive to avoid division by zero and ensure a meaningful candle.
            if candle_range > 0:
                candle_body_ratio = candle_body / candle_range
                if candle_body_ratio >= 0.6: # Candle Body Ratio >= 60%
                    is_bullish_momentum = True
        
        if not is_bullish_momentum:
            continue

        # --- Condition 2: Pullback from the Strong Bullish Candle's Close ---
        # Current price must be below the close of the strong bullish candle.
        if current_price >= prev_candle.close:
            continue # Not a pullback, or price has already recovered/exceeded

        # --- Condition 3: Relative Pullback Depth (relaxed: 0% to 80% of strong candle's body) ---
        # Price must not have fallen below the open of the strong bullish candle.
        pullback_amount = prev_candle.close - current_price
        
        # `candle_body` was calculated in Condition 1 and is guaranteed > 0 here because `is_bullish_momentum` is true.
        if candle_body <= 0: # Defensive check, though should be covered by is_bullish_momentum
            continue

        if current_price < prev_candle.open_price:
            continue # Pullback is too deep, below the open of the bullish candle

        relative_pullback_depth_ratio = pullback_amount / candle_body
        if not (0.0 < relative_pullback_depth_ratio <= 0.8): # Allow pullbacks between 0% and 80%
            continue

        # --- Condition 4: Stochastic Oscillator (relaxed oversold: %K and %D below 60, %K > %D) ---
        stoch_period = 14
        stoch_d_sma_period = 3

        closes = [c.close for c in pair_data.warm]
        highs = [c.high for c in pair_data.warm]
        lows = [c.low for c in pair_data.warm]
        
        all_stoch_k_values = []
        for i in range(len(pair_data.warm)):
            k = _calculate_stoch_k(closes, highs, lows, stoch_period, i)
            all_stoch_k_values.append(k)

        # `len(all_stoch_k_values)` is `len(pair_data.warm)`, which is at least MIN_WARM_CANDLES_FOR_STOCH (15).
        # So, `[-stoch_d_sma_period:]` will be safe (i.e., `[-3:]`).
        stoch_k = all_stoch_k_values[-1] # The most recent %K value
        stoch_d = np.mean(all_stoch_k_values[-stoch_d_sma_period:]) # 3-period SMA of %K

        # Relaxed oversold condition: K and D below 60, K is above D (or crossing up)
        if not (stoch_k < 60 and stoch_d < 60 and stoch_k > stoch_d):
            continue

        # --- Condition 5: Volatility and Spread Filters (relaxed/broadened) ---
        # Average Bid-Ask Spread
        # `spread_rel` in Tick model is already a percentage (e.g., 0.1 for 0.1%). Convert to ratio.
        current_spread_rel_ratio = current_tick.spread_rel / 100.0
        if current_spread_rel_ratio >= 0.005: # Spread must be less than 0.5% (0.005 as ratio)
            continue

        # Hot Realized Volatility
        # Calculate log returns for the most recent ticks (e.g., last 60 ticks = 1 minute)
        # `pair_data.hot` is guaranteed to have at least MIN_HOT_TICKS_FOR_VOL (2 ticks) due to initial check.
        vol_lookback_ticks = min(len(pair_data.hot), 60) # Use up to 60 recent ticks for volatility
        prices_for_vol = [t.last_price for t in pair_data.hot[-vol_lookback_ticks:]]

        real_vol = 0.0
        if len(prices_for_vol) >= 2:
            log_returns = [math.log(prices_for_vol[i] / prices_for_vol[i-1]) for i in range(1, len(prices_for_vol))]
            if log_returns: # Ensure there's at least one return to calculate std dev
                real_vol = np.std(log_returns)
        
        if real_vol >= 0.02: # Realized Volatility must be less than 2% (0.02 as ratio)
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