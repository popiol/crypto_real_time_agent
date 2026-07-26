from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal

RULE_ID = "SMA_Cross_12_Exploratory"
SMA_PERIOD = 12
PROFIT_TARGET_MULTIPLIER = 1.02
STOP_LOSS_MULTIPLIER = 0.99
MIN_WARM_CANDLES = SMA_PERIOD
MIN_HOT_TICKS = 2

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm data for SMA calculation
        if len(pair_data.warm) < MIN_WARM_CANDLES:
            continue

        # Calculate SMA(12) from the last 12 hourly close prices
        closes_for_sma = [candle.close for candle in pair_data.warm[-SMA_PERIOD:]]
        sma_12 = sum(closes_for_sma) / SMA_PERIOD

        # Ensure enough hot data for current and previous price comparison
        if len(pair_data.hot) < MIN_HOT_TICKS:
            continue

        # Get the current and previous last prices from hot data
        current_tick = pair_data.hot[-1]
        previous_tick = pair_data.hot[-2]
        current_price = current_tick.last_price
        previous_price = previous_tick.last_price

        # Entry Condition: Price crosses above SMA(12)
        # We check previous price against SMA_12 and current price against SMA_12
        # to detect a crossover.
        if previous_price <= sma_12 and current_price > sma_12:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=current_tick.polled_at,
                    price=current_price,
                    rule_id=RULE_ID,
                    entry_price=current_price,
                    take_profit_multiplier=PROFIT_TARGET_MULTIPLIER,
                    stop_loss_multiplier=STOP_LOSS_MULTIPLIER,
                    indicators={"sma_12": sma_12}
                )
            )
        # Exit Condition: Price crosses below SMA(12)
        # This condition is used to close an existing long position.
        elif previous_price >= sma_12 and current_price < sma_12:
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=current_tick.polled_at,
                    price=current_price,
                    rule_id=RULE_ID,
                    exit_reason="SMA_Bearish_Crossover",
                    indicators={"sma_12": sma_12}
                )
            )

    return signals