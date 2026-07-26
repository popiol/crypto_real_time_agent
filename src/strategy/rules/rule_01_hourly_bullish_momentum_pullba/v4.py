from __future__ import annotations
import math
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, Tick, WarmCandle

RULE_ID = "relaxed_bullish_pullback_entry_v1"

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Initial Data Check:
        # Stochastic Oscillator and Z-score require 14 warm candles.
        # Volatility and Spread calculations require 100 hot ticks.
        if len(pair_data.warm) < 14 or len(pair_data.hot) < 100:
            continue

        last_candle = pair_data.warm[-1]
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_timestamp = current_tick.polled_at

        # --- Condition 1: Bullish Momentum (Candle Body Ratio) ---
        candle_body_ratio = 0.0
        candle_range = last_candle.high - last_candle.low
        if candle_range > 0:
            candle_body_ratio = abs(last_candle.close - last_candle.open_price) / candle_range
        # If candle_range is 0, candle_body_ratio remains 0.0 as per pseudocode.

        is_bullish_momentum = candle_body_ratio > 0.4 and last_candle.close > last_candle.open_price

        # --- Condition 2: Shallow Pullback (Price Z-Score) ---
        closes_14 = [c.close for c in pair_data.warm[-14:]]
        # len(closes_14) is guaranteed to be 14 due to initial data check.
        
        mean_14 = sum(closes_14) / len(closes_14)
        
        std_dev_14 = 0.0
        # Calculate population standard deviation as per pseudocode.
        # This handles the case where std_dev_14 would be 0 if all closes were identical.
        if len(closes_14) > 0: # Should always be true (length 14)
            std_dev_14 = (sum((x - mean_14)**2 for x in closes_14) / len(closes_14))**0.5

        price_z_score = 0.0
        if std_dev_14 > 0: # Avoid division by zero if prices are flat
            price_z_score = (current_price - mean_14) / std_dev_14
        # If std_dev_14 is 0, price_z_score remains 0.0 as per pseudocode.
        
        is_shallow_pullback = -1.5 <= price_z_score <= -0.1

        # --- Condition 3: Stable Market (Hot Realized Volatility & Average Bid-Ask Spread) ---
        recent_hot_ticks = pair_data.hot[-100:]
        hot_prices = [t.last_price for t in recent_hot_ticks]
        
        hot_volatility = 0.0
        if len(hot_prices) >= 2:
            returns = []
            for i in range(1, len(hot_prices)):
                if hot_prices[i-1] != 0: # Avoid division by zero
                    returns.append((hot_prices[i] - hot_prices[i-1]) / hot_prices[i-1])
            
            if returns: # Ensure returns list is not empty before calculation
                # Calculate population standard deviation of returns as per pseudocode.
                mean_returns = sum(returns) / len(returns)
                hot_volatility = (sum((r - mean_returns)**2 for r in returns) / len(returns))**0.5
        # If len(hot_prices) < 2 or returns is empty, hot_volatility remains 0.0.
        
        avg_spread = 0.0
        valid_spreads = []
        for tick in recent_hot_ticks:
            # Ensure bid/ask prices are valid before calculating spread.
            if tick.bid_price is not None and tick.ask_price is not None and tick.bid_price > 0 and tick.ask_price > 0:
                valid_spreads.append(tick.ask_price - tick.bid_price)
        
        if valid_spreads:
            avg_spread = sum(valid_spreads) / len(valid_spreads)
        # If no valid spreads, avg_spread remains 0.0 as per pseudocode logic.

        is_stable_market = (0.00005 < hot_volatility < 0.0025) and \
                           (0.00005 < avg_spread < 0.0015)

        # --- Condition 4: Stochastic Oscillator ---
        highs_14 = [c.high for c in pair_data.warm[-14:]]
        lows_14 = [c.low for c in pair_data.warm[-14:]]

        lowest_low = min(lows_14)
        highest_high = max(highs_14)

        stoch_k = 50.0 # Default value if range is zero, as per pseudocode
        if (highest_high - lowest_low) > 0:
            stoch_k = ((last_candle.close - lowest_low) / (highest_high - lowest_low)) * 100
        
        is_oversold_stoch = stoch_k < 45
        is_overbought_stoch = stoch_k > 85

        # --- Signal Generation ---
        # Buy Signal: All entry conditions must be met.
        if is_bullish_momentum and is_shallow_pullback and is_stable_market and is_oversold_stoch:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))
        # Sell Signal: Exit existing long position if Stochastic becomes overbought.
        # This is an exit condition, not a short entry.
        elif is_overbought_stoch:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))

    return signals