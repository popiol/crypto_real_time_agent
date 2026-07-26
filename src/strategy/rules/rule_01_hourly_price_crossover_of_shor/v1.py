from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal

# Rule ID for this specific trading rule
RULE_ID = "hourly_sma_crossover_1"

# The period for the Simple Moving Average
SMA_PERIOD = 8

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates buy/sell signals based on the hourly price crossing an 8-hour Simple Moving Average (SMA).

    A buy signal is triggered when the current live price crosses above the SMA from below.
    A sell signal is triggered when the current live price crosses below the SMA from above.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for SMA calculation (at least SMA_PERIOD candles)
        if len(pair_data.warm) < SMA_PERIOD:
            continue

        # Ensure we have at least one hot tick for the current live price
        if not pair_data.hot:
            continue

        # data.warm is ordered from newest to oldest.
        # We need the close prices of the last SMA_PERIOD completed hourly candles.
        # pair_data.warm[:SMA_PERIOD] gives us the most recent SMA_PERIOD candles.
        hourly_closes = [candle.close for candle in pair_data.warm[:SMA_PERIOD]]

        # Calculate the SMA for the last SMA_PERIOD hours
        sma_value = statistics.mean(hourly_closes)

        # Get the most recent live tick data
        current_tick = pair_data.hot[0]
        current_price = current_tick.last_price
        current_timestamp = current_tick.polled_at

        # Get the close price of the most recently completed hourly candle.
        # This is used to determine if a crossover has occurred.
        previous_hourly_close = pair_data.warm[0].close

        # --- Buy Signal Condition ---
        # A buy signal occurs if the previous hourly close was at or below the SMA,
        # and the current live price is now above the SMA.
        if previous_hourly_close <= sma_value and current_price > sma_value:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=RULE_ID,
                # Stop loss set slightly below the SMA, take profit slightly above current price
                stop_loss=sma_value * 0.995,
                take_profit=current_price * 1.015,
                indicators={"sma_8": sma_value}
            ))
        # --- Sell Signal Condition ---
        # A sell signal occurs if the previous hourly close was at or above the SMA,
        # and the current live price is now below the SMA.
        elif previous_hourly_close >= sma_value and current_price < sma_value:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=RULE_ID,
                # Stop loss set slightly above the SMA, take profit slightly below current price
                stop_loss=sma_value * 1.005,
                take_profit=current_price * 0.985,
                indicators={"sma_8": sma_value}
            ))

    return signals