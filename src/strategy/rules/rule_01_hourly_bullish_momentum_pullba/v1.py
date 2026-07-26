from __future__ import annotations
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

# Constants
# PROFIT_TARGET_PERCENT and STOP_LOSS_PERCENT are not used by the rule's internal SellSignal logic
# as they are assumed to be handled by the agent based on fixed configurations.
# They are included here only because they were in the original pseudocode.
PROFIT_TARGET_PERCENT = 0.015
STOP_LOSS_PERCENT = 0.0075
PULLBACK_MAX_PERCENT = 0.005  # Max 0.5% pullback from last hourly close for entry
STRONG_CANDLE_MIN_BODY_PERCENT = 0.01  # Min 1% body size for strong candle (close - open / open)

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    rule_id = "hourly_bullish_pullback_entry"

    for pair, pair_data in data.items():
        # Ensure sufficient data for analysis
        # Need at least 2 warm candles for second_last_candle and last_candle
        # Need at least 1 hot tick for current_price and timestamp
        if len(pair_data.warm) < 2 or not pair_data.hot:
            continue

        last_candle = pair_data.warm[-1]
        second_last_candle = pair_data.warm[-2]
        current_price = pair_data.hot[-1].last_price
        current_timestamp = pair_data.hot[-1].polled_at

        # Validate last_candle.open_price to prevent division by zero
        if last_candle.open_price == 0:
            continue

        # --- Buy Signal Logic ---

        # Calculate candle body size as a percentage of its open
        last_candle_body_size = abs(last_candle.close - last_candle.open_price)
        last_candle_body_percent = last_candle_body_size / last_candle.open_price

        # Condition 1: Last hourly candle was strongly bullish
        # It closed higher than it opened, higher than the previous candle's close,
        # and its body size meets the minimum percentage.
        is_strong_bullish_candle = (
            last_candle.close > last_candle.open_price and
            last_candle.close > second_last_candle.close and
            last_candle_body_percent >= STRONG_CANDLE_MIN_BODY_PERCENT
        )

        # Condition 2: Current price is in a slight pullback from the strong candle's close
        # It must be below the last candle's close but above its open, within a defined pullback range.
        is_pullback_entry = (
            current_price < last_candle.close and
            current_price >= (last_candle.close * (1 - PULLBACK_MAX_PERCENT)) and
            current_price > last_candle.open_price
        )

        if is_strong_bullish_candle and is_pullback_entry:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=rule_id,
                confidence=1.0
            ))

        # --- Sell Signal Logic (to close an existing long position) ---
        # As per constraints, profit target and stop loss are handled by the agent.
        # The rule provides a SellSignal only if its underlying bullish hypothesis is invalidated
        # by a significant market reversal.
        # The pseudocode's "Reversal below the strong candle's open" is interpreted here
        # as a general market condition: if the current price falls below the open of the
        # *most recent* completed hourly candle, it indicates a strong bearish reversal,
        # invalidating any current long position based on recent bullish momentum.
        if current_price < last_candle.open_price:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=rule_id,
                confidence=1.0
            ))

    return signals