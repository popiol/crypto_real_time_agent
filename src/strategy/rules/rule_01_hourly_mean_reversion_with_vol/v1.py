from __future__ import annotations
# No 'statistics' module needed as sum/len is used for averages
from src.agent.models import BuySignal, MarketData, SellSignal

# Rule ID as per the idea description
RULE_ID = "mean_reversion_vol_band_v1"
# Lookback period for SMA and ATR
PERIOD = 5
# Multiplier for ATR band
MULTIPLIER = 1.5

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Ensure enough warm candle data for SMA(PERIOD) and ATR(PERIOD).
        # ATR(PERIOD) requires PERIOD True Ranges. Each True Range for candle_i
        # depends on candle_i and candle_{i-1}'s close.
        # To calculate PERIOD True Ranges for the last PERIOD candles (e.g., warm_candles[N-PERIOD] to warm_candles[N-1]),
        # we need access to warm_candles[N-PERIOD-1] for the first TR calculation.
        # Thus, len(warm_candles) must be at least PERIOD + 1.
        if len(warm_candles) < PERIOD + 1:
            continue

        # Ensure at least one hot tick for the current price
        if not hot_ticks:
            continue

        # 1. Calculate SMA(PERIOD)
        # Using the last 'PERIOD' candles for SMA.
        closes = [candle.close for candle in warm_candles[-PERIOD:]]
        sma_value = sum(closes) / PERIOD

        # 2. Calculate ATR(PERIOD)
        true_ranges = []
        # Iterate over the last 'PERIOD' candles to calculate their True Ranges.
        # The loop starts from `len(warm_candles) - PERIOD` to `len(warm_candles) - 1`.
        # For each candle `i`, we use `warm_candles[i]` and `warm_candles[i-1].close`.
        for i in range(len(warm_candles) - PERIOD, len(warm_candles)):
            high = warm_candles[i].high
            low = warm_candles[i].low
            prev_close = warm_candles[i-1].close

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        atr_value = sum(true_ranges) / PERIOD

        # 3. Get current price and timestamp from the latest hot tick
        current_tick = hot_ticks[-1]
        current_price = current_tick.last_price
        polled_at = current_tick.polled_at

        # 4. Calculate volatility band thresholds
        buy_threshold = sma_value - (MULTIPLIER * atr_value)
        sell_threshold = sma_value + (MULTIPLIER * atr_value)

        # 5. Generate signals based on current price deviation from the band
        if current_price < buy_threshold:
            signals.append(BuySignal(
                pair=pair,
                timestamp=polled_at,
                price=current_price,
                rule_id=RULE_ID,
            ))
        elif current_price > sell_threshold:
            signals.append(SellSignal(
                pair=pair,
                timestamp=polled_at,
                price=current_price,
                rule_id=RULE_ID,
            ))
    return signals