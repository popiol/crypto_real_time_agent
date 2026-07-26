from __future__ import annotations
import math
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, Tick, WarmCandle

RULE_ID = "relaxed_bullish_pullback_entry_v1"

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # 1. Initial Data Check for sufficient history
        # Stochastic Oscillator requires 14 warm candles.
        # Volatility and Spread calculations require 60 hot ticks.
        if len(pair_data.warm) < 14 or len(pair_data.hot) < 60:
            continue

        last_hourly_candle = pair_data.warm[-1]
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_timestamp = current_tick.polled_at

        # Calculate pullback_midpoint, used for both buy and sell (invalidation) logic.
        # This is the midpoint of the strong bullish candle's body.
        pullback_midpoint = (last_hourly_candle.open_price + last_hourly_candle.close) / 2

        # --- Condition 1: Relaxed Strong Bullish Candle ---
        # Checks if the last hourly candle shows significant bullish strength.
        is_strong_bullish_candle = False
        # Price gain of at least 0.4%
        if last_hourly_candle.close > last_hourly_candle.open_price * 1.004:
            candle_range = last_hourly_candle.high - last_hourly_candle.low
            candle_body = last_hourly_candle.close - last_hourly_candle.open_price
            # Candle body must constitute at least 55% of its total range.
            if candle_range > 0 and candle_body / candle_range > 0.55:
                is_strong_bullish_candle = True

        # SellSignal (invalidation logic):
        # If a strong bullish candle was identified, but the current price has dropped
        # below the acceptable pullback midpoint, it invalidates the bullish thesis.
        # This signals to close any existing long position opened by this rule.
        if is_strong_bullish_candle and current_price < pullback_midpoint:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence for an invalidation exit
            ))
            continue # If an invalidation occurs, no buy signal for this pair.

        # If a strong bullish candle is not present, we cannot proceed with a buy signal.
        if not is_strong_bullish_candle:
            continue

        # --- Condition 2: Relaxed Calm Pullback ---
        # Checks if the current price has pulled back slightly but remains above the pullback midpoint.
        is_calm_pullback = False
        # At this point, `is_strong_bullish_candle` is true and `current_price >= pullback_midpoint`
        # (due to the SellSignal check above). So we only need to check the upper bound of the pullback.
        if current_price < last_hourly_candle.close:
            is_calm_pullback = True

        # If not a calm pullback (e.g., price is still above the strong candle's close), no buy signal.
        if not is_calm_pullback:
            continue

        # --- Condition 3: Relaxed Acceptable Volatility and Spread ---
        # Assesses market conditions based on recent tick data.
        is_acceptable_market_conditions = False
        
        recent_hot_ticks = pair_data.hot[-60:]
        hot_prices = [tick.last_price for tick in recent_hot_ticks]
        
        hot_spreads_values = []
        for tick in recent_hot_ticks:
            # Ensure bid/ask prices are valid before calculating spread.
            if tick.bid_price is not None and tick.ask_price is not None and tick.bid_price > 0 and tick.ask_price > 0:
                hot_spreads_values.append(tick.ask_price - tick.bid_price)

        # Volatility Calculation: Root Mean Square of Returns.
        hot_realized_volatility = 0.0
        if len(hot_prices) >= 2:
            returns = []
            for i in range(1, len(hot_prices)):
                if hot_prices[i-1] != 0: # Avoid division by zero.
                    returns.append((hot_prices[i] - hot_prices[i-1]) / hot_prices[i-1])
            
            if returns: # Ensure returns list is not empty before calculation.
                hot_realized_volatility = (sum(r**2 for r in returns) / len(returns))**0.5

        # Average Bid-Ask Spread Calculation.
        avg_bid_ask_spread = 0.0
        if len(hot_spreads_values) > 0:
            avg_bid_ask_spread = sum(s for s in hot_spreads_values) / len(hot_spreads_values)

        # Market conditions are acceptable if volatility and spread are within defined, relaxed ranges.
        # Both volatility and spread must be greater than 0, indicating some market activity, but not excessive.
        if (hot_realized_volatility > 0 and hot_realized_volatility < 0.0006 and
            avg_bid_ask_spread > 0 and avg_bid_ask_spread < 0.0004):
            is_acceptable_market_conditions = True

        if not is_acceptable_market_conditions:
            continue # No buy signal if market conditions are not acceptable.

        # --- Condition 4: Stochastic Oscillator Confirmation ---
        # Confirms oversold conditions during the pullback with a relaxed threshold.
        is_stoch_confirm = False
        
        # Gather 14 periods of close, low, and high prices from warm candles.
        closes_14 = [c.close for c in pair_data.warm[-14:]]
        lows_14 = [c.low for c in pair_data.warm[-14:]]
        highs_14 = [c.high for c in pair_data.warm[-14:]]

        lowest_low_14 = min(lows_14)
        highest_high_14 = max(highs_14)

        stoch_k = 0.0
        # Calculate Stochastic %K. Avoid division by zero if range is flat.
        if (highest_high_14 - lowest_low_14) > 0:
            stoch_k = ((closes_14[-1] - lowest_low_14) / (highest_high_14 - lowest_low_14)) * 100

        # Stochastic %K below 40 indicates a relaxed oversold condition.
        if stoch_k < 40:
            is_stoch_confirm = True

        if not is_stoch_confirm:
            continue # No buy signal if Stochastic is not confirming.

        # --- Final Buy Signal Generation ---
        # If all conditions (strong bullish candle, calm pullback, acceptable market conditions,
        # and Stochastic confirmation) are met, generate a BuySignal.
        signals.append(BuySignal(
            pair=pair,
            timestamp=current_timestamp,
            price=current_price,
            rule_id=RULE_ID,
            confidence=1.0 # Standard confidence for a triggered rule.
        ))

    return signals