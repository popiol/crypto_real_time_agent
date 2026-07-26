from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal

RULE_ID = "SMA_12_Crossover_Debug"
SMA_PERIOD = 12
PROFIT_TARGET_MULTIPLIER = 1.02
STOP_LOSS_MULTIPLIER = 0.99
MIN_WARM_CANDLES = SMA_PERIOD
MIN_HOT_TICKS = 1 # Only the most recent tick is needed for current_price

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm data for SMA calculation
        if len(pair_data.warm) < MIN_WARM_CANDLES:
            continue

        # Ensure enough hot data for current price
        if not pair_data.hot:
            continue

        # Calculate SMA(12) from the last 12 hourly close prices
        # data.warm contains hourly candles, index 0 is the oldest, -1 is the newest
        closes_for_sma = [candle.close for candle in pair_data.warm[-SMA_PERIOD:]]
        sma_12 = sum(closes_for_sma) / SMA_PERIOD

        current_price = pair_data.hot[-1].last_price # Most recent tick price
        previous_candle_close = pair_data.warm[-1].close # Close of the last completed hourly candle
        current_tick_timestamp = pair_data.hot[-1].polled_at

        # Buy Signal: Current price crosses above SMA(12)
        # Condition: current_price is above SMA and previous_candle_close was below SMA
        if current_price > sma_12 and previous_candle_close < sma_12:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=current_tick_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    entry_price=current_price,
                    take_profit_multiplier=PROFIT_TARGET_MULTIPLIER,
                    stop_loss_multiplier=STOP_LOSS_MULTIPLIER,
                    indicators={"sma_12": sma_12}
                )
            )
        # Sell Signal: Current price crosses below SMA(12) (to close an existing long position)
        # Condition: current_price is below SMA and previous_candle_close was above SMA
        elif current_price < sma_12 and previous_candle_close > sma_12:
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=current_tick_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    exit_reason="SMA_Bearish_Crossover",
                    indicators={"sma_12": sma_12}
                )
            )

    return signals