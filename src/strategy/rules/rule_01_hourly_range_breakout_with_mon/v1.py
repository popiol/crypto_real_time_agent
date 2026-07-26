from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal

RULE_ID = "hourly-range-breakout-monthly-filter-001"
MIN_WARM_CANDLES = 8
MIN_HOT_TICKS = 1
MIN_COLD_MONTHS = 1

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient data for calculations
        if (
            len(pair_data.hot) < MIN_HOT_TICKS
            or len(pair_data.warm) < MIN_WARM_CANDLES
            or len(pair_data.cold) < MIN_COLD_MONTHS
        ):
            continue

        current_price = pair_data.hot[-1].last_price
        timestamp = pair_data.hot[-1].polled_at
        monthly_avg_price = pair_data.cold[0].avg_price # data.cold[0] is assumed to be the current month's data

        # Calculate highest high and lowest low over the last 8 hourly candles
        last_8_candles = pair_data.warm[-MIN_WARM_CANDLES:]
        highest_high_8h = max(candle.high for candle in last_8_candles)
        lowest_low_8h = min(candle.low for candle in last_8_candles)

        # Long condition
        if current_price > highest_high_8h and current_price > monthly_avg_price:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    reason="Hourly high breakout with monthly bullish trend",
                )
            )

        # Short condition
        if current_price < lowest_low_8h and current_price < monthly_avg_price:
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    reason="Hourly low breakout with monthly bearish trend",
                )
            )
            
    return signals