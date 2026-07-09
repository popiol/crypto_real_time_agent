"""Rule 09 — ROC Momentum Alignment Simplified."""
from __future__ import annotations
from src.agent.models import BuySignal, MarketData, PairData, SellSignal

# Parameters from the rule idea's pseudocode
SHORT_ROC_PERIOD = 5  # e.g., 5 periods for short-term ROC
MEDIUM_ROC_PERIOD = 20  # e.g., 20 periods for medium-term ROC
ROC_THRESHOLD = 0.001  # A small threshold to define "positive" or "negative" momentum, to avoid noise around zero

# Minimum data requirements for calculations
# For short-term ROC: need current price and price 'SHORT_ROC_PERIOD' periods ago.
# This requires at least SHORT_ROC_PERIOD + 1 ticks.
MIN_TICKS = SHORT_ROC_PERIOD + 1

# For medium-term ROC: need current candle close and close 'MEDIUM_ROC_PERIOD' periods ago.
# This requires at least MEDIUM_ROC_PERIOD + 1 warm candles.
MIN_WARM_CANDLES = MEDIUM_ROC_PERIOD + 1


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        warm_candles = pair_data.warm

        # Ensure we have enough data points to calculate both ROCs
        if len(ticks) < MIN_TICKS or len(warm_candles) < MIN_WARM_CANDLES:
            continue

        # --- Calculate short-term ROC using tick data ---
        # Current price is the last tick's price
        current_tick_price = ticks[-1].last_price
        # Price 'SHORT_ROC_PERIOD' periods ago
        price_short_ago = ticks[-SHORT_ROC_PERIOD - 1].last_price

        # Avoid division by zero if historical price is zero (unlikely for real assets)
        if price_short_ago == 0:
            continue

        roc_short = (current_tick_price - price_short_ago) / price_short_ago

        # --- Calculate medium-term ROC using warm candle data ---
        # Current price is the last warm candle's close
        current_candle_price = warm_candles[-1].close
        # Price 'MEDIUM_ROC_PERIOD' periods ago
        price_medium_ago = warm_candles[-MEDIUM_ROC_PERIOD - 1].close

        # Avoid division by zero
        if price_medium_ago == 0:
            continue

        roc_medium = (current_candle_price - price_medium_ago) / price_medium_ago

        # --- Signal Generation ---
        ts = ticks[-1].polled_at
        price = current_tick_price

        # Buy signal: Both short-term and medium-term ROC are positive (above threshold)
        if roc_short > ROC_THRESHOLD and roc_medium > ROC_THRESHOLD:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=price))
        # Sell signal: Both short-term and medium-term ROC are negative (below negative threshold)
        elif roc_short < -ROC_THRESHOLD and roc_medium < -ROC_THRESHOLD:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=price))

    return signals