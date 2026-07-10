"""Rule 02 — Bollinger Band Mean-Reversion with Volume Spike Confirmation (v1)."""
from __future__ import annotations
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle

# Bollinger Band parameters
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0

# Volume Spike parameters
VOLUME_LOOKBACK_TICKS = 10  # Number of previous ticks to average volume over
VOLUME_SPIKE_MULTIPLIER = 1.5  # Current volume must be this much higher than average

# Minimum data requirements
MIN_CANDLES_FOR_BB = BB_PERIOD
# Need VOLUME_LOOKBACK_TICKS for the average and 1 for the current tick
MIN_TICKS_FOR_VOLUME = VOLUME_LOOKBACK_TICKS + 1 

RULE_ID = "rule_02_bollinger_band_volume_v1"


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm candle data for Bollinger Bands calculation
        if len(pair_data.warm) < MIN_CANDLES_FOR_BB:
            continue

        # Ensure sufficient hot tick data for volume spike calculation
        if len(pair_data.hot) < MIN_TICKS_FOR_VOLUME:
            continue
        
        # Ensure VOLUME_LOOKBACK_TICKS is positive to calculate a meaningful average
        if VOLUME_LOOKBACK_TICKS <= 0:
            continue

        # --- Calculate Bollinger Bands ---
        # Use the last BB_PERIOD warm candles for calculation
        bb_closes = [c.close for c in pair_data.warm[-BB_PERIOD:]]
        
        sma = statistics.mean(bb_closes)
        
        # Handle cases where standard deviation cannot be calculated (e.g., all prices are the same)
        # statistics.stdev requires at least 2 data points.
        # If BB_PERIOD is 1, len(bb_closes) would be 1, so we handle it here.
        # If BB_PERIOD >= 2, this check is redundant due to MIN_CANDLES_FOR_BB.
        if len(bb_closes) < 2:
             continue
        
        try:
            std = statistics.stdev(bb_closes)
        except statistics.StatisticsError:
            # This happens if all values in bb_closes are identical, meaning std dev is 0.
            std = 0.0

        if std == 0.0:
            # If standard deviation is zero, bands collapse, so no meaningful signal.
            continue

        upper_band = sma + (std * BB_STD_DEV_MULTIPLIER)
        lower_band = sma - (std * BB_STD_DEV_MULTIPLIER)

        # --- Calculate Volume Spike ---
        # Get the current tick and its 24-hour rolling volume
        current_tick: Tick = pair_data.hot[-1]
        current_volume: float = current_tick.volume_24h

        # Get the 24-hour rolling volumes from the previous VOLUME_LOOKBACK_TICKS,
        # excluding the current tick. This ensures the average is not skewed by the potential spike itself.
        previous_volumes: list[float] = [
            t.volume_24h for t in pair_data.hot[-MIN_TICKS_FOR_VOLUME:-1]
        ]
        
        # Calculate the average of these previous volumes
        avg_volume_previous = statistics.mean(previous_volumes)

        # Check for volume spike condition
        volume_spike = False
        if avg_volume_previous > 0:
            # Current volume is significantly higher than its recent average
            volume_spike = current_volume > (avg_volume_previous * VOLUME_SPIKE_MULTIPLIER)
        elif current_volume > 0:
            # If the average of previous volumes was zero (e.g., no trading activity),
            # any positive current volume indicates a spike.
            volume_spike = True

        # --- Generate Signals ---
        current_price = current_tick.last_price
        ts = current_tick.polled_at

        if current_price < lower_band and volume_spike:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price, rule_id=RULE_ID))
        elif current_price > upper_band and volume_spike:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price, rule_id=RULE_ID))

    return signals