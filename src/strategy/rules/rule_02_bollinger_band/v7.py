"""Rule 02 — Mean reversion: Bollinger Band with Volume Confirmation."""
from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, PairData, SellSignal

K = 2.0
BB_PERIOD_CANDLES = 10  # Period for Bollinger Band calculation using warm candles (hourly)
VOLUME_SMA_PERIOD_TICKS = 20 # Period for Volume SMA calculation using hot ticks (per-poll)
VOLUME_CONFIRMATION_FACTOR = 1.5 # Multiplier for volume confirmation threshold (e.g., 1.5 * V_SMA)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure there's at least one hot tick for current price/time/volume access
        if not pair_data.hot:
            continue

        # --- Bollinger Band Calculation (using warm candles, as in original rule_02) ---
        if len(pair_data.warm) < BB_PERIOD_CANDLES:
            continue

        # Use the most recent 'BB_PERIOD_CANDLES' for calculation
        closes = [c.close for c in pair_data.warm[-BB_PERIOD_CANDLES:]]
        
        # Calculate mean and standard deviation for Bollinger Bands
        mean_price = statistics.mean(closes)
        
        # Handle cases where std dev might be zero (e.g., all closes are identical)
        # In such cases, bands would collapse, and no meaningful signal can be generated.
        if len(closes) < 2 or statistics.stdev(closes) == 0:
            std_dev_price = 0.0
        else:
            std_dev_price = statistics.stdev(closes)

        if std_dev_price == 0:
            continue

        bb_upper = mean_price + K * std_dev_price
        bb_lower = mean_price - K * std_dev_price

        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # --- Volume Confirmation Calculation (using hot ticks) ---
        if len(pair_data.hot) < VOLUME_SMA_PERIOD_TICKS:
            continue

        # Use the 'volume_24h' from the most recent 'VOLUME_SMA_PERIOD_TICKS'
        volumes_24h = [t.volume_24h for t in pair_data.hot[-VOLUME_SMA_PERIOD_TICKS:]]
        
        # Ensure there's enough data for volume calculation and prevent zero division
        # If all recent volumes are zero, then volume confirmation cannot be met.
        if not volumes_24h or all(v == 0 for v in volumes_24h):
            continue
        
        volume_sma = statistics.mean(volumes_24h)
        
        # This check is technically redundant if `all(v == 0 for v in volumes_24h)` passed,
        # but provides an explicit safety net if `statistics.mean` somehow returns 0 unexpectedly
        # with non-zero inputs (which it shouldn't).
        if volume_sma == 0:
            continue

        current_volume_24h = pair_data.hot[-1].volume_24h
        volume_confirmation_threshold = VOLUME_CONFIRMATION_FACTOR * volume_sma

        # --- Signal Generation with Volume Confirmation ---
        # Buy signal: Price drops below lower BB AND current volume is significantly higher than average
        if current_price < bb_lower and current_volume_24h > volume_confirmation_threshold:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Sell signal: Price rises above upper BB AND current volume is significantly higher than average
        elif current_price > bb_upper and current_volume_24h > volume_confirmation_threshold:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals