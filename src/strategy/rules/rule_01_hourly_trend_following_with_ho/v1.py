from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal

# Constants for the rule
SMA_PERIOD = 10
HOT_TICK_LOOKBACK = 30
SPREAD_THRESHOLD_REL = 0.001 # 0.1%

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # --- Data Preparation ---
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Check for sufficient warm data for SMA(10)
        if len(warm_candles) < SMA_PERIOD:
            continue # Not enough hourly data for SMA(10)

        # Calculate SMA(10) from the last 10 hourly closes
        hourly_closes = [candle.close for candle in warm_candles]
        sma_10_hourly = sum(hourly_closes[-SMA_PERIOD:]) / SMA_PERIOD

        # Check for sufficient hot data
        if not hot_ticks:
            continue # No hot tick data available

        current_tick = hot_ticks[-1]
        current_last_price = current_tick.last_price
        current_bid_price = current_tick.bid_price
        current_ask_price = current_tick.ask_price
        current_spread = current_ask_price - current_bid_price
        
        # --- Buy Signal Logic ---
        buy_condition_met = False

        # Condition 1: Hourly uptrend
        if current_last_price > sma_10_hourly:
            # Condition 2: Short-term breakout (requires enough hot data)
            # We need at least (HOT_TICK_LOOKBACK + 1) ticks to get the last tick and the previous HOT_TICK_LOOKBACK ticks
            if len(hot_ticks) > HOT_TICK_LOOKBACK: 
                # Get last_price for the preceding HOT_TICK_LOOKBACK ticks (excluding the very last one)
                recent_hot_prices_for_breakout = [tick.last_price for tick in hot_ticks[-(HOT_TICK_LOOKBACK + 1):-1]]
                
                if recent_hot_prices_for_breakout: # Ensure the list is not empty after slicing
                    max_price_prev_30_ticks = max(recent_hot_prices_for_breakout)

                    if current_last_price > max_price_prev_30_ticks:
                        # Condition 3: Tight spread
                        if current_last_price > 0 and current_spread < (SPREAD_THRESHOLD_REL * current_last_price):
                            buy_condition_met = True
            
        if buy_condition_met:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_tick.polled_at,
                price=current_last_price,
                reason="Hourly uptrend confirmed, short-term breakout, and tight spread"
            ))

        # --- Sell Signal Logic (to close existing long positions) ---
        # This rule generates a SellSignal when the hourly uptrend reverses.
        if current_last_price < sma_10_hourly:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_tick.polled_at,
                price=current_last_price,
                reason="Hourly uptrend reversal detected"
            ))

    return signals