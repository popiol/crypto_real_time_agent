from __future__ import annotations

from src.agent.models import BuySignal, MarketData, SellSignal

# Minimum number of hourly candles required to calculate both current and previous 8-hour SMAs.
# To calculate the current 8-hour SMA, we need the last 8 candles.
# To calculate the previous 8-hour SMA, we need the 8 candles immediately preceding the last one.
# Therefore, a total of 8 (for previous) + 1 (for current) = 9 candles are needed.
MIN_CANDLES_FOR_SMA_CROSS = 9


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough data for 8-hour SMA crossover detection
        if len(warm_candles) < MIN_CANDLES_FOR_SMA_CROSS:
            continue

        # Extract closing prices for current SMAs (latest candles)
        # current_sma8 uses the last 8 candles
        # current_sma3 uses the last 3 candles
        current_closes_8 = [candle.close for candle in warm_candles[-8:]]
        current_closes_3 = [candle.close for candle in warm_candles[-3:]]

        current_sma3 = sum(current_closes_3) / 3
        current_sma8 = sum(current_closes_8) / 8

        # Extract closing prices for previous SMAs (candles immediately before the last one)
        # previous_sma8 uses the 8 candles from index -9 up to -2
        # previous_sma3 uses the 3 candles from index -4 up to -2
        previous_closes_8 = [candle.close for candle in warm_candles[-9:-1]]
        previous_closes_3 = [candle.close for candle in warm_candles[-4:-1]]

        previous_sma3 = sum(previous_closes_3) / 3
        previous_sma8 = sum(previous_closes_8) / 8

        # Get the latest candle for signal timestamp and price
        latest_candle = warm_candles[-1]

        # Check for BUY signal: 3-hour SMA crosses above 8-hour SMA
        if current_sma3 > current_sma8 and previous_sma3 <= previous_sma8:
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_candle.hour,
                price=latest_candle.close,
                rule_id="hourly_sma_cross_001",
                indicators={
                    "current_sma3": current_sma3,
                    "current_sma8": current_sma8,
                    "previous_sma3": previous_sma3,
                    "previous_sma8": previous_sma8,
                }
            ))
        # Check for SELL signal: 3-hour SMA crosses below 8-hour SMA
        elif current_sma3 < current_sma8 and previous_sma3 >= previous_sma8:
            signals.append(SellSignal(
                pair=pair,
                timestamp=latest_candle.hour,
                price=latest_candle.close,
                rule_id="hourly_sma_cross_001",
                indicators={
                    "current_sma3": current_sma3,
                    "current_sma8": current_sma8,
                    "previous_sma3": previous_sma3,
                    "previous_sma8": previous_sma8,
                }
            ))

    return signals