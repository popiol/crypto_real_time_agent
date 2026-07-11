from __future__ import annotations

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


# Proposed parameter adjustments
NEW_SHORT_ROC_PERIOD = 10
NEW_MEDIUM_ROC_PERIOD = 30
NEW_SHORT_ROC_THRESHOLD = 0.005
NEW_MEDIUM_ROC_THRESHOLD = 0.002
NEW_NEUTRAL_MOMENTUM_LOWER_BOUND = -0.002
NEW_NEUTRAL_MOMENTUM_UPPER_BOUND = 0.002

# Minimum data requirements
# For Short_Term_ROC: we need prices[-1] and prices[-NEW_SHORT_ROC_PERIOD - 1]
MIN_TICKS = NEW_SHORT_ROC_PERIOD + 1
# For Medium_Term_ROC: we need closes[-1] and closes[-NEW_MEDIUM_ROC_PERIOD - 1]
# For Previous_Medium_Term_ROC_Value: we need closes[-2] and closes[-NEW_MEDIUM_ROC_PERIOD - 2]
# Thus, we need at least NEW_MEDIUM_ROC_PERIOD + 2 warm candles.
MIN_WARM_CANDLES = NEW_MEDIUM_ROC_PERIOD + 2


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        warm_candles = pair_data.warm

        if len(ticks) < MIN_TICKS or len(warm_candles) < MIN_WARM_CANDLES:
            continue

        # Extract prices for short-term ROC calculation from ticks
        prices = [t.last_price for t in ticks]
        
        # Extract close prices for medium-term ROC calculation from warm candles
        closes = [c.close for c in warm_candles]

        # Calculate Short_Term_ROC
        # (Current_Price - Price_SHORT_ROC_PERIOD_Ago) / Price_SHORT_ROC_PERIOD_Ago
        price_current_tick = prices[-1]
        price_short_roc_period_ago = prices[-NEW_SHORT_ROC_PERIOD - 1]
        
        # Avoid division by zero
        if price_short_roc_period_ago == 0:
            continue
        short_term_roc = (price_current_tick - price_short_roc_period_ago) / price_short_roc_period_ago

        # Calculate Medium_Term_ROC
        # (Current_Price - Price_MEDIUM_ROC_PERIOD_Ago) / Price_MEDIUM_ROC_PERIOD_Ago
        price_current_candle_close = closes[-1]
        price_medium_roc_period_ago_candle_close = closes[-NEW_MEDIUM_ROC_PERIOD - 1]
        
        # Avoid division by zero
        if price_medium_roc_period_ago_candle_close == 0:
            continue
        medium_term_roc = (price_current_candle_close - price_medium_roc_period_ago_candle_close) / price_medium_roc_period_ago_candle_close

        # Calculate Previous_Medium_Term_ROC_Value
        # This is interpreted as the Medium_Term_ROC for the candle immediately preceding the current one.
        # (Price_SHORT_ROC_PERIOD_Ago - Price_MEDIUM_ROC_PERIOD_Ago_Shifted) / Price_MEDIUM_ROC_PERIOD_Ago_Shifted
        # Re-interpreting for candles: (closes[-2] - closes[-NEW_MEDIUM_ROC_PERIOD - 2]) / closes[-NEW_MEDIUM_ROC_PERIOD - 2]
        price_prev_candle_close = closes[-2]
        price_prev_medium_roc_period_ago_candle_close = closes[-NEW_MEDIUM_ROC_PERIOD - 2]
        
        # Avoid division by zero
        if price_prev_medium_roc_period_ago_candle_close == 0:
            continue
        previous_medium_term_roc_value = (price_prev_candle_close - price_prev_medium_roc_period_ago_candle_close) / price_prev_medium_roc_period_ago_candle_close

        ts = ticks[-1].polled_at
        price = prices[-1]

        # Apply the new signal generation conditions
        if (short_term_roc > NEW_SHORT_ROC_THRESHOLD and
                medium_term_roc > NEW_MEDIUM_ROC_THRESHOLD and
                previous_medium_term_roc_value < NEW_NEUTRAL_MOMENTUM_UPPER_BOUND):
            signals.append(BuySignal(pair=pair, timestamp=ts, price=price))
        elif (short_term_roc < -NEW_SHORT_ROC_THRESHOLD and
              medium_term_roc < -NEW_MEDIUM_ROC_THRESHOLD and
              previous_medium_term_roc_value > NEW_NEUTRAL_MOMENTUM_LOWER_BOUND):
            signals.append(SellSignal(pair=pair, timestamp=ts, price=price))

    return signals