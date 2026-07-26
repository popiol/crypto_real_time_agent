from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal

# Constants
SMA_PERIOD = 5  # Hourly candles
DEVIATION_THRESHOLD = 0.001  # 0.1% deviation
MAX_SPREAD_BPS = 5  # Max spread in basis points (0.05%)
MIN_TICKS_FOR_CURRENT_DATA = 1 # We need at least one tick for current price, bid, ask, spread

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    
    for pair, pair_data in data.items():
        # Data checks for warm candles (SMA)
        if len(pair_data.warm) < SMA_PERIOD:
            continue # Not enough data for SMA

        # Data checks for hot ticks (current price, spread)
        if len(pair_data.hot) < MIN_TICKS_FOR_CURRENT_DATA:
            continue # Not enough current tick data

        # Calculate 5-period SMA from data.warm hourly closes
        # Ensure we only take the last SMA_PERIOD candles
        hourly_closes = [candle.close for candle in pair_data.warm[-SMA_PERIOD:]]
        sma_5 = sum(hourly_closes) / SMA_PERIOD

        # Get current market data from the latest tick
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_bid = current_tick.bid_price
        current_ask = current_tick.ask_price
        current_spread = current_tick.spread_abs # Using spread_abs from Tick model

        # Check spread condition (convert MAX_SPREAD_BPS to absolute value based on current price)
        # Avoid division by zero if current_price is zero, though unlikely for real trading
        if current_price <= 0:
            continue
            
        max_allowed_spread = current_price * (MAX_SPREAD_BPS / 10000)

        if current_spread > max_allowed_spread:
            continue # Spread too wide, avoid trading

        # Check deviation from SMA
        deviation = (current_price - sma_5) / sma_5

        if deviation < -DEVIATION_THRESHOLD:
            # Price is significantly below SMA, expect reversion upwards
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_tick.polled_at,
                price=current_bid,
                rule_id="mean_reversion_spread_filter_v1",
                reason="Price significantly below SMA, low spread"
            ))
        elif deviation > DEVIATION_THRESHOLD:
            # Price is significantly above SMA, expect reversion downwards
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_tick.polled_at,
                price=current_ask,
                rule_id="mean_reversion_spread_filter_v1",
                reason="Price significantly above SMA, low spread"
            ))
            
    return signals