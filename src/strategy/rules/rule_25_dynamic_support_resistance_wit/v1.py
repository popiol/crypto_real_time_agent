from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

# --- Configuration Constants ---
# Number of previous periods (ticks) for Average Trading Volume (ATV) calculation.
# This is a proxy as WarmCandle lacks volume data.
VOLUME_LOOKBACK_PERIOD = 20

# Multiplier for ATV to determine the volume threshold for confirmation.
VOLUME_MULTIPLIER = 1.5

# Minimum warm candles required to calculate Pivot Points (PP), Support (S1), and Resistance (R1).
# We need at least 2 candles: one for previous HLC, one for current.
MIN_CANDLES_FOR_SR = 2

# Minimum ticks required to calculate Average Trading Volume (ATV) and current volume.
# This requires VOLUME_LOOKBACK_PERIOD previous ticks + 1 for the current tick.
MIN_TICKS_FOR_VOLUME = VOLUME_LOOKBACK_PERIOD + 1

# Rule ID as provided in the idea description
RULE_ID = "a0f4a02b-bdc7-4138-b5dd-e351c04520f5"


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        ticks = pair_data.hot

        # 1. Handle insufficient data gracefully
        # Ensure enough warm candle data for S/R calculation
        if len(warm_candles) < MIN_CANDLES_FOR_SR:
            continue

        # Ensure enough tick data for volume calculation
        if len(ticks) < MIN_TICKS_FOR_VOLUME:
            continue

        # --- Data extraction for current period ---
        # The 'current' candle is the most recent complete candle (warm_candles[-1]).
        # The 'previous' candle is the one before that (warm_candles[-2]).
        current_candle = warm_candles[-1]
        prev_candle = warm_candles[-2]

        # The 'current' tick is the most recent real-time tick (ticks[-1]).
        # Used for timestamp, last price, and current volume.
        current_tick = ticks[-1]

        # --- Step 1: Calculate Pivot Points (PP), Support 1 (S1), Resistance 1 (R1) ---
        # Based on the previous period's High, Low, Close.
        high_prev = prev_candle.high
        low_prev = prev_candle.low
        close_prev = prev_candle.close

        pp = (high_prev + low_prev + close_prev) / 3
        r1 = (2 * pp) - low_prev
        s1 = (2 * pp) - high_prev

        # --- Step 2 & 3: Calculate Average Trading Volume (ATV) and Volume Threshold ---
        # IMPORTANT ASSUMPTION: The WarmCandle model does not include volume data.
        # As a workaround, we use Tick.volume_24h as a proxy for current and historical
        # volume activity. This is a simplification due to data model limitations.
        current_volume = current_tick.volume_24h

        # Calculate ATV from previous ticks' 24h rolling volume.
        # We take `VOLUME_LOOKBACK_PERIOD` previous ticks, excluding the very last one (current_tick).
        historical_volumes_24h = [t.volume_24h for t in ticks[-(VOLUME_LOOKBACK_PERIOD + 1):-1]]

        if not historical_volumes_24h:
            continue # Should not happen if MIN_TICKS_FOR_VOLUME check passes, but good for robustness

        atv = statistics.mean(historical_volumes_24h)

        # Avoid division by zero or nonsensical thresholds if ATV is zero or negative.
        if atv <= 0:
            continue

        volume_threshold = atv * VOLUME_MULTIPLIER
        volume_confirmed = current_volume > volume_threshold

        # --- Step 4: Buy Signal ---
        buy_signal_generated = False
        if volume_confirmed:
            # Condition 1: Price approaches S1 and rebounds with increasing volume
            # This implies the price touched or went below S1, then closed above S1,
            # and the current close is higher than the previous close, showing upward momentum.
            if (current_candle.low <= s1 and
                current_candle.close > s1 and
                current_candle.close > prev_candle.close):
                signals.append(BuySignal(
                    pair=pair,
                    timestamp=current_tick.polled_at, # Use current tick timestamp for precision
                    price=current_tick.last_price,
                    rule_id=RULE_ID
                ))
                buy_signal_generated = True

            # Condition 2: Price breaks above R1 with high volume
            # The current candle closed above R1, having previously been at or below R1.
            elif (current_candle.close > r1 and
                  prev_candle.close <= r1):
                signals.append(BuySignal(
                    pair=pair,
                    timestamp=current_tick.polled_at,
                    price=current_tick.last_price,
                    rule_id=RULE_ID
                ))
                buy_signal_generated = True

        # --- Step 5: Sell Signal ---
        # Only check for a sell signal if no buy signal was generated for the current period
        # to avoid conflicting signals on the same candle.
        if not buy_signal_generated and volume_confirmed:
            # Condition 1: Price approaches R1 and rejects with increasing volume
            # This implies the price touched or went above R1, then closed below R1,
            # and the current close is lower than the previous close, showing downward momentum.
            if (current_candle.high >= r1 and
                current_candle.close < r1 and
                current_candle.close < prev_candle.close):
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=current_tick.polled_at,
                    price=current_tick.last_price,
                    rule_id=RULE_ID
                ))

            # Condition 2: Price breaks below S1 with high volume
            # The current candle closed below S1, having previously been at or above S1.
            elif (current_candle.close < s1 and
                  prev_candle.close >= s1):
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=current_tick.polled_at,
                    price=current_tick.last_price,
                    rule_id=RULE_ID
                ))

    return signals