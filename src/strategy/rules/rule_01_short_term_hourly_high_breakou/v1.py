from __future__ import annotations
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from datetime import datetime

# Rule specific constants
MIN_WARM_CANDLES_FOR_LOOKBACK = 5
RULE_ID = "hourly_high_breakout_v1"
CONFIDENCE_LEVEL = 1.0 # Default confidence for a direct rule trigger

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candle data for the 5-hour lookback
        if len(pair_data.warm) < MIN_WARM_CANDLES_FOR_LOOKBACK:
            continue

        # Ensure we have at least one hot tick for the current price
        if not pair_data.hot:
            continue

        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        timestamp = current_tick.polled_at

        # Get the closing prices of the last N hourly candles
        # data.warm is typically sorted oldest to newest, so slicing from the end
        # gets the most recent candles.
        historical_closes = [
            candle.close for candle in pair_data.warm[-MIN_WARM_CANDLES_FOR_LOOKBACK:]
        ]

        # Find the highest closing price among these historical candles
        max_close_past_5h = max(historical_closes)

        # Buy Condition: Current price breaks above the highest close of the past 5 hours
        # The 'NOT position_open' check from the pseudocode is handled by the trading system
        # which only opens a position if one isn't already active for the pair.
        if current_price > max_close_past_5h:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=CONFIDENCE_LEVEL,
                )
            )

        # Sell Condition: The rule describes a stop-loss based on `entry_price`.
        # However, the `signal` function operates without knowledge of open positions
        # or their entry prices (NO POSITION VISIBILITY constraint).
        # Therefore, this specific stop-loss condition cannot be implemented by the rule itself.
        # Position exits (including stop-losses) are handled by the broader trading system,
        # which may use its own configured stop-loss or max hold time.
        # This rule only generates BuySignals based on the breakout condition.

    return signals