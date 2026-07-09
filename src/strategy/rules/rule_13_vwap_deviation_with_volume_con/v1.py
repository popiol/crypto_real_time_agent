import statistics
from datetime import datetime
from typing import List, Optional, Union

from src.agent.models import BuySignal, MarketData, SellSignal, Tick

# Rule ID for identification
RULE_ID = "6c966287-f17c-40f2-b261-d3ffc4eb3f89"

# Configuration parameters from pseudocode
VWAP_PERIOD = 20
STD_DEV_PERIOD = 20
VOLUME_AVG_PERIOD = 50
DEVIATION_THRESHOLD_MULTIPLIER = 2.0  # Corresponds to `deviation_threshold`
VOLUME_THRESHOLD_MULTIPLIER = 1.5    # Corresponds to `volume_threshold`

# Minimum ticks required for all calculations.
# Max of all lookback periods ensures we have enough data for all calculations.
MIN_TICKS_REQUIRED = max(VWAP_PERIOD, STD_DEV_PERIOD, VOLUME_AVG_PERIOD)


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


def _calculate_std_dev_from_vwap(
    prices: List[float], vwap: float, std_dev_period: int
) -> Optional[float]:
    """
    Calculates the standard deviation of price deviations from a given VWAP
    over a specified lookback period.
    """
    if len(prices) < std_dev_period:
        return None

    # Take the most recent `std_dev_period` prices
    period_prices = prices[-std_dev_period:]

    deviations = [p - vwap for p in period_prices]

    if len(deviations) > 1:
        return statistics.stdev(deviations)
    elif len(deviations) == 1:
        return 0.0  # Standard deviation of a single point is 0
    else:
        return None  # Should not happen if len(prices) check passed


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


def signal(data: MarketData) -> List[Union[BuySignal, SellSignal]]:
    """
    Detects when the price significantly deviates from its Volume Weighted Average Price (VWAP),
    confirmed by high trading volume, indicating strong directional conviction.
    Emits a Buy signal when price is significantly below VWAP and volume is high.
    Emits a Sell signal when price is significantly above VWAP and volume is high.
    """
    signals: List[Union[BuySignal, SellSignal]] = []

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

        # 2. Calculate standard deviation of price from the calculated VWAP
        std_dev_from_vwap = _calculate_std_dev_from_vwap(prices, vwap, STD_DEV_PERIOD)
        if std_dev_from_vwap is None:
            continue

        # 3. Calculate average volume
        avg_volume = _calculate_average_volume(volumes, VOLUME_AVG_PERIOD)
        if avg_volume is None:
            continue

        # Define thresholds for deviation and volume
        lower_band = vwap - DEVIATION_THRESHOLD_MULTIPLIER * std_dev_from_vwap
        upper_band = vwap + DEVIATION_THRESHOLD_MULTIPLIER * std_dev_from_vwap
        volume_confirmation_threshold = VOLUME_THRESHOLD_MULTIPLIER * avg_volume

        # Generate signals based on conditions
        # Buy signal: price significantly below VWAP AND high volume
        if current_price < lower_band and current_volume > volume_confirmation_threshold:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=current_tick.polled_at,
                    price=current_price,
                    rule_id=RULE_ID,
                )
            )
        # Sell signal: price significantly above VWAP AND high volume
        elif (
            current_price > upper_band
            and current_volume > volume_confirmation_threshold
        ):
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=current_tick.polled_at,
                    price=current_price,
                    rule_id=RULE_ID,
                )
            )

    return signals