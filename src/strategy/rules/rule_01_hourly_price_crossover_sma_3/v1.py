from __future__ import annotations
from datetime import datetime
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

RULE_ID = "SMA_3_Cross_V1"
SMA_PERIOD = 3
MIN_CANDLES_FOR_CROSSOVER = SMA_PERIOD + 1 # Need SMA_PERIOD candles for current SMA, and 1 more for previous price/SMA comparison

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        hourly_candles = pair_data.warm

        # Ensure enough data for SMA(3) and a crossover check (at least 4 candles)
        if len(hourly_candles) < MIN_CANDLES_FOR_CROSSOVER:
            continue

        # Extract closing prices for the required lookback
        # We need at least the last 4 closes for current and previous SMA(3) and prices
        closes = [c.close for c in hourly_candles[-MIN_CANDLES_FOR_CROSSOVER:]]

        current_price = closes[-1]
        previous_price = closes[-2]

        # Calculate current SMA(3) using the last 3 closes
        current_sma_3 = sum(closes[-SMA_PERIOD:]) / SMA_PERIOD

        # Calculate previous SMA(3) using the 3 closes *before* the current one
        # This uses closes[-4], closes[-3], closes[-2]
        previous_sma_3 = sum(closes[-(SMA_PERIOD+1):-1]) / SMA_PERIOD
        
        # The timestamp and price for the signal should be from the most recent candle
        signal_timestamp = hourly_candles[-1].hour
        signal_price = hourly_candles[-1].close # Use close price for signals as per the rule idea

        # Buy Signal: Current price crosses above SMA(3)
        # This means previous_price was at or below previous_sma_3, and current_price is above current_sma_3
        if previous_price <= previous_sma_3 and current_price > current_sma_3:
            signals.append(BuySignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
                rule_id=RULE_ID,
                confidence=1.0 # Fixed confidence for this basic rule
            ))

        # Sell Signal: Current price crosses below SMA(3)
        # This means previous_price was at or above previous_sma_3, and current_price is below current_sma_3
        elif previous_price >= previous_sma_3 and current_price < current_sma_3:
            signals.append(SellSignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
                rule_id=RULE_ID,
                confidence=1.0 # Fixed confidence for this basic rule
            ))

    return signals