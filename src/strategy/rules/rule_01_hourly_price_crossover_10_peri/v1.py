from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal

SMA_PERIOD = 10

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm data for the SMA calculation
        if len(pair_data.warm) < SMA_PERIOD:
            continue

        # Ensure we have hot data for the current price and timestamp
        if not pair_data.hot:
            continue

        # Calculate the 10-period Simple Moving Average (SMA) from warm data
        # We need the last SMA_PERIOD completed hourly candles.
        sma_closes = [candle.close for candle in pair_data.warm[-SMA_PERIOD:]]
        sma_10 = sum(sma_closes) / SMA_PERIOD

        # Get the current price from the latest tick in hot data
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price

        # Get the close price of the most recently completed hourly candle
        previous_close = pair_data.warm[-1].close

        # Buy signal: current price crosses above SMA(10)
        # Confirmed if the previous close was below or equal to the SMA
        if current_price > sma_10 and previous_close <= sma_10:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_tick.polled_at,
                price=current_price,
            ))
        # Sell signal: current price crosses below SMA(10)
        # Confirmed if the previous close was above or equal to the SMA
        elif current_price < sma_10 and previous_close >= sma_10:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_tick.polled_at,
                price=current_price,
            ))

    return signals