from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    RULE_ID = "sma_crossover_5_20_hourly"
    
    # Define SMA periods
    SMA_SHORT_PERIOD = 5
    SMA_LONG_PERIOD = 20

    # Minimum candles needed for both current and previous long SMA
    # To calculate current 20-SMA, we need 20 candles.
    # To calculate previous 20-SMA, we need the 20 candles *before* the current one.
    # So, to have both current and previous, we need at least 20 (for current) + 1 (for the "previous" state) = 21 candles.
    MIN_CANDLES_REQUIRED = SMA_LONG_PERIOD + 1

    for pair, pair_data in data.items():
        candles = pair_data.warm.candles
        
        if len(candles) < MIN_CANDLES_REQUIRED:
            # Not enough data to calculate both current and previous SMAs
            continue

        # Extract closing prices
        closes = [c.close for c in candles]

        # Get the latest candle for signal timestamp and price
        latest_candle: WarmCandle = candles[-1]
        signal_timestamp: datetime = latest_candle.hour
        signal_price: float = latest_candle.close

        # Calculate current SMAs (using the last N candles)
        sma_short_current = sum(closes[-SMA_SHORT_PERIOD:]) / SMA_SHORT_PERIOD
        sma_long_current = sum(closes[-SMA_LONG_PERIOD:]) / SMA_LONG_PERIOD

        # Calculate previous SMAs (using candles ending one period ago)
        # This means excluding the very last candle from the calculation.
        sma_short_prev = sum(closes[-(SMA_SHORT_PERIOD + 1):-1]) / SMA_SHORT_PERIOD
        sma_long_prev = sum(closes[-(SMA_LONG_PERIOD + 1):-1]) / SMA_LONG_PERIOD

        # Buy Signal: Short SMA crosses above Long SMA
        # Current short SMA is above current long SMA, AND
        # Previous short SMA was below or equal to previous long SMA
        if sma_short_current > sma_long_current and sma_short_prev <= sma_long_prev:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=signal_timestamp,
                    price=signal_price,
                    rule_id=RULE_ID,
                    confidence=1.0, # High confidence for a direct crossover signal
                )
            )
        
        # Sell Signal: Short SMA crosses below Long SMA
        # Current short SMA is below current long SMA, AND
        # Previous short SMA was above or equal to previous long SMA
        # This is intended to close existing long positions.
        elif sma_short_current < sma_long_current and sma_short_prev >= sma_long_prev:
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=signal_timestamp,
                    price=signal_price,
                    rule_id=RULE_ID,
                    confidence=1.0, # High confidence for a direct crossover signal
                )
            )

    return signals