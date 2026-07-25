from __future__ import annotations
import statistics
from datetime import datetime
from typing import List

from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Minimum number of hourly candles required to calculate both current and previous 
# 5-period and 20-period SMAs for crossover detection.
# To calculate the current 20-period SMA, we need 20 candles.
# To calculate the previous 20-period SMA (using data up to the second-to-last candle),
# we need 20 candles from the `closes[:-1]` list.
# This means `len(closes) - 1 >= 20`, so `len(closes) >= 21`.
MIN_CANDLES_FOR_SMAS = 21

RULE_ID = "DailySMACrossover_5_20"

def _calculate_sma(prices: List[float], period: int) -> float:
    """
    Calculates the Simple Moving Average for the given period.
    Assumes `prices` list is long enough for the period.
    """
    return statistics.mean(prices[-period:])

def signal(data: MarketData) -> List[BuySignal | SellSignal]:
    signals: List[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # The rule's description specifies "daily close prices".
        # However, `data.warm` provides at most 24 hourly candles, and `data.cold` is monthly.
        # To implement a 5-period and 20-period SMA crossover with the available granularity
        # and sufficient lookback, we adapt to use hourly close prices from `data.warm`.
        # This aligns with the rule's rationale for using "higher frequency data" for responsiveness.
        
        # Ensure candles are sorted by time to correctly calculate SMAs
        warm_candles: List[WarmCandle] = sorted(pair_data.warm, key=lambda c: c.hour)
        closes = [c.close for c in warm_candles]

        if len(closes) < MIN_CANDLES_FOR_SMAS:
            # Not enough historical hourly candles to calculate both current and previous
            # 5-period and 20-period SMAs for reliable crossover detection.
            continue

        # Get the latest candle's details for signal generation
        latest_candle = warm_candles[-1]
        signal_timestamp = latest_candle.hour
        signal_price = latest_candle.close

        # Calculate current SMAs using the full list of `closes`
        sma_5_current = _calculate_sma(closes, 5)
        sma_20_current = _calculate_sma(closes, 20)

        # Calculate previous SMAs using data up to the second-to-last candle (`closes[:-1]`)
        closes_prev = closes[:-1]
        sma_5_prev = _calculate_sma(closes_prev, 5)
        sma_20_prev = _calculate_sma(closes_prev, 20)

        # Check for BUY signal: 5-period SMA crosses above 20-period SMA
        # This occurs if current SMA_5 is above SMA_20, AND previous SMA_5 was at or below SMA_20.
        if sma_5_current > sma_20_current and sma_5_prev <= sma_20_prev:
            signals.append(BuySignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
                rule_id=RULE_ID
            ))
        # Check for SELL signal: 5-period SMA crosses below 20-period SMA
        # This occurs if current SMA_5 is below SMA_20, AND previous SMA_5 was at or above SMA_20.
        elif sma_5_current < sma_20_current and sma_5_prev >= sma_20_prev:
            signals.append(SellSignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
                rule_id=RULE_ID
            ))

    return signals