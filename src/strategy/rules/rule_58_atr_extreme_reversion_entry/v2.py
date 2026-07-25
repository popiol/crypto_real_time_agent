from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

# Rule specific constants
RULE_ID = "ATR-Excursion-Threshold-Relax-V1"
ATR_PERIOD = 14
LOW_HIGH_PERIOD = 24
ATR_MULTIPLIER = 1.0  # Key change: Reduced multiplier for relaxed entry

def _calculate_true_range(high: float, low: float, prev_close: float) -> float:
    """Calculates the True Range for a single candle."""
    return max(high - low, abs(high - prev_close), abs(low - prev_close))

def _calculate_atr(candles: list[WarmCandle], period: int) -> float:
    """
    Calculates the Simple Moving Average of True Range (ATR) over a given period.
    Requires at least (period + 1) candles to calculate 'period' true ranges.
    """
    if len(candles) < period + 1:
        return 0.0

    true_ranges = []
    # Extract the relevant candles for ATR calculation.
    # We need the last `period + 1` candles.
    relevant_candles = candles[-(period + 1):]

    for i in range(1, len(relevant_candles)):
        tr = _calculate_true_range(
            high=relevant_candles[i].high,
            low=relevant_candles[i].low,
            prev_close=relevant_candles[i-1].close
        )
        true_ranges.append(tr)
    
    if len(true_ranges) < period:
        return 0.0

    # Calculate SMA of the `period` true ranges
    return float(np.mean(true_ranges))

def _get_period_low(candles: list[WarmCandle], period: int) -> float:
    """
    Returns the lowest *low price* over the last 'period' candles.
    Uses 'low' field of WarmCandle.
    """
    if len(candles) < period:
        # Return a very high value if insufficient data, so it won't trigger a long signal prematurely.
        return float('inf') 
    
    # Get low prices for the last 'period' candles
    low_prices = [c.low for c in candles[-period:]]
    return min(low_prices)

def _get_period_high(candles: list[WarmCandle], period: int) -> float:
    """
    Returns the highest *high price* over the last 'period' candles.
    Uses 'high' field of WarmCandle.
    """
    if len(candles) < period:
        # Return a very low value if insufficient data, so it won't trigger a short signal prematurely.
        return float('-inf') 
    
    # Get high prices for the last 'period' candles
    high_prices = [c.high for c in candles[-period:]]
    return max(high_prices)

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Minimum data required:
        # 1. LOW_HIGH_PERIOD (24) candles for calculating 24-period low/high.
        # 2. ATR_PERIOD + 1 (15) candles for calculating 14-period ATR.
        # The maximum of these is 24, so we need at least 24 warm candles.
        if len(warm_candles) < LOW_HIGH_PERIOD:
            continue

        # Need at least one hot tick to get the current price
        if not hot_ticks:
            continue

        current_price = hot_ticks[-1].last_price
        
        atr_14 = _calculate_atr(warm_candles, ATR_PERIOD)
        
        # If ATR is zero, it implies insufficient data for ATR or no volatility.
        # In either case, we should avoid generating signals.
        if atr_14 == 0.0:
            continue

        low_24 = _get_period_low(warm_candles, LOW_HIGH_PERIOD)
        high_24 = _get_period_high(warm_candles, LOW_HIGH_PERIOD)

        long_entry_threshold = low_24 - (ATR_MULTIPLIER * atr_14)
        short_entry_threshold = high_24 + (ATR_MULTIPLIER * atr_14)

        # Long entry condition: current price drops below 24-period low by more than 1 * ATR
        if current_price < long_entry_threshold:
            signals.append(BuySignal(
                pair=pair,
                timestamp=hot_ticks[-1].polled_at,  # Use the timestamp of the latest tick
                price=current_price,
                rule_id=RULE_ID,
            ))
        # Short entry condition: current price rises above 24-period high by more than 1 * ATR
        elif current_price > short_entry_threshold:
            signals.append(SellSignal(
                pair=pair,
                timestamp=hot_ticks[-1].polled_at,  # Use the timestamp of the latest tick
                price=current_price,
                rule_id=RULE_ID,
            ))
            
    return signals