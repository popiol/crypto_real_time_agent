from __future__ import annotations
import statistics
from datetime import datetime
from typing import List, Optional, Union

from src.agent.models import BuySignal, MarketData, SellSignal, Tick

# Rule ID for identification - Using the idea_id as the rule_id for this specific modification
RULE_ID = "f3ff1509-d2bb-47af-9202-1a4d70571a46"

# Configuration parameters from pseudocode
VWAP_PERIOD = 20
VOLUME_MA_PERIOD = 50
VOLUME_MULTIPLIER = 1.5
DEVIATION_THRESHOLD = 0.005  # Direct percentage deviation from VWAP

# Minimum ticks required for all calculations.
# Max of all lookback periods ensures we have enough data for all calculations.
MIN_TICKS_REQUIRED = max(VWAP_PERIOD, VOLUME_MA_PERIOD)


def _calculate_vwap(
    prices: List[float], volumes: List[float], lookback_period: int
) -> Optional[float]:
    """
    Calculates the Volume Weighted Average Price (VWAP) for the given lookback period.
    Assumes `prices` and `volumes` are ordered lists, with the most recent data at the end.
    """
    if len(prices) < lookback_period or len(volumes) < lookback_period:
        return None

    # Take the most recent `lookback_period` data points
    period_prices = prices[-lookback_period:]
    period_volumes = volumes[-lookback_period:]

    vwap_sum = sum(p * v for p, v in zip(period_prices, period_volumes))
    volume_sum = sum(period_volumes)

    if volume_sum > 0:
        return vwap_sum / volume_sum
    else:
        # If no volume in the period, VWAP is undefined.
        return None


def _calculate_average_volume(
    volumes: List[float], lookback_period: int
) -> Optional[float]:
    """
    Calculates the average volume over a specified lookback period.
    """
    if len(volumes) < lookback_period:
        return None

    period_volumes = volumes[-lookback_period:]

    if len(period_volumes) > 0:
        return statistics.mean(period_volumes)
    else:
        return None


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Detects significant price deviations from the Volume Weighted Average Price (VWAP)
    that are confirmed by a sustained increase in trading volume, relative to recent average volume.
    Emits a Buy signal when the price is significantly below VWAP with volume exceeding
    a multiple of its recent average, and a Sell signal when the price is significantly
    above VWAP with similarly elevated volume.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot

        # Ensure we have enough data for all lookback periods
        if len(ticks) < MIN_TICKS_REQUIRED:
            continue

        # Extract relevant data for the current pair
        # `volume_24h` is used as the proxy for volume in the absence of per-tick volume.
        prices = [t.last_price for t in ticks]
        volumes = [t.volume_24h for t in ticks]

        current_tick = ticks[-1]
        current_price = current_tick.last_price
        current_volume = current_tick.volume_24h

        # 1. Calculate VWAP for the current point
        vwap = _calculate_vwap(prices, volumes, VWAP_PERIOD)
        if vwap is None:
            continue

        # 2. Calculate average volume
        avg_volume = _calculate_average_volume(volumes, VOLUME_MA_PERIOD)
        if avg_volume is None:
            continue

        # Calculate price deviation as a percentage from VWAP
        # This replaces the STD_DEV based deviation from the previous version.
        price_deviation = (current_price - vwap) / vwap

        # Check for volume confirmation
        # This confirms if current volume is significantly higher than recent average.
        is_volume_confirmed = current_volume > (avg_volume * VOLUME_MULTIPLIER)

        # Generate signals based on conditions
        # Buy signal: price significantly below VWAP AND volume is confirmed
        if price_deviation < -DEVIATION_THRESHOLD and is_volume_confirmed:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=current_tick.polled_at,
                    price=current_price,
                    rule_id=RULE_ID,
                )
            )
        # Sell signal: price significantly above VWAP AND volume is confirmed
        elif price_deviation > DEVIATION_THRESHOLD and is_volume_confirmed:
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=current_tick.polled_at,
                    price=current_price,
                    rule_id=RULE_ID,
                )
            )

    return signals