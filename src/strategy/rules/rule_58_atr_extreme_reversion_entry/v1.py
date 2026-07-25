from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Rule specific constants
RULE_ID = "atr-extreme-reversion-v1"
ATR_PERIOD = 14
LOW_HIGH_PERIOD = 24
ATR_MULTIPLIER = 2.0  # For entry condition

def _calculate_true_range(high: float, low: float, prev_close: float) -> float:
    """Calculates the True Range for a single candle."""
    return max(high - low, abs(high - prev_close), abs(low - prev_close))

def _calculate_atr(candles: list[WarmCandle], period: int) -> float:
    """
    Calculates the Simple Moving Average of True Range (ATR) over a given period.
    Requires at least (period + 1) candles to calculate 'period' true ranges.
    """
    # We need `period` true range values, each requiring the current and previous candle's close.
    # So, for `period` TRs, we need `period + 1` candles in total.
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
    
    # After the loop, `true_ranges` should contain exactly `period` values if `len(relevant_candles)` was `period + 1`.
    if len(true_ranges) < period:
        # This case should ideally not be hit if the initial `len(candles) < period + 1` check is correct.
        return 0.0

    # Calculate SMA of the `period` true ranges
    return float(np.mean(true_ranges))

def _get_period_low(candles: list[WarmCandle], period: int) -> float:
    """Returns the lowest close price over the last 'period' candles."""
    if len(candles) < period:
        # Return a very high value if insufficient data, so it won't trigger a long signal prematurely.
        return float('inf') 
    
    # Get close prices for the last 'period' candles
    close_prices = [c.close for c in candles[-period:]]
    return min(close_prices)

def _get_period_high(candles: list[WarmCandle], period: int) -> float:
    """Returns the highest close price over the last 'period' candles."""
    if len(candles) < period:
        # Return a very low value if insufficient data, so it won't trigger a short signal prematurely.
        return float('-inf') 
    
    # Get close prices for the last 'period' candles
    close_prices = [c.close for c in candles[-period:]]
    return max(close_prices)

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Minimum data required:
        # 1. LOW_HIGH_PERIOD candles for calculating 24-period low/high.
        # 2. ATR_PERIOD + 1 candles for calculating 14-period ATR.
        # Since LOW_HIGH_PERIOD (24) is greater than ATR_PERIOD + 1 (15),
        # we need at least 24 candles to proceed.
        if len(warm_candles) < LOW_HIGH_PERIOD:
            continue

        current_close = warm_candles[-1].close
        
        atr_14 = _calculate_atr(warm_candles, ATR_PERIOD)
        
        # If ATR is zero, it implies insufficient data for ATR or no volatility.
        # In either case, we should avoid generating signals.
        if atr_14 == 0.0:
            continue

        low_24 = _get_period_low(warm_candles, LOW_HIGH_PERIOD)
        high_24 = _get_period_high(warm_candles, LOW_HIGH_PERIOD)

        # Long entry condition: current close drops below 24-period low minus 2 * ATR
        if current_close < (low_24 - (ATR_MULTIPLIER * atr_14)):
            signals.append(BuySignal(
                pair=pair,
                timestamp=warm_candles[-1].hour,  # Use the close time of the last candle
                price=current_close,
                rule_id=RULE_ID,
                # Stop-loss and take-profit are strategy management parameters,
                # not part of the signal object based on the provided model.
            ))
        # Short entry condition: current close rises above 24-period high plus 2 * ATR
        elif current_close > (high_24 + (ATR_MULTIPLIER * atr_14)):
            signals.append(SellSignal(
                pair=pair,
                timestamp=warm_candles[-1].hour,  # Use the close time of the last candle
                price=current_close,
                rule_id=RULE_ID,
                # Stop-loss and take-profit are strategy management parameters,
                # not part of the signal object based on the provided model.
            ))
            
    return signals