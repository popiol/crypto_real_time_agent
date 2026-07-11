from __future__ import annotations
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Rule configuration
CMF_PERIOD = 20
BUY_THRESHOLD = 0.10
SELL_THRESHOLD = -0.10
RULE_ID = "e1608ce9-58a2-4061-ae37-8c1b40febb10"

def _calculate_cmf_for_window(window_high: list[float], window_low: list[float], window_close: list[float], window_volume: list[float]) -> float:
    """
    Calculates the Chaikin Money Flow (CMF) for a given window of price and volume data.
    """
    mfv_sum = 0.0
    volume_sum = 0.0

    for i in range(len(window_close)):
        mf_multiplier = 0.0
        high_low_diff = window_high[i] - window_low[i]

        # Avoid division by zero if high and low are the same
        if high_low_diff != 0:
            # Money Flow Multiplier
            mf_multiplier = ((window_close[i] - window_low[i]) - (window_high[i] - window_close[i])) / high_low_diff
        
        # Money Flow Volume
        mfv = mf_multiplier * window_volume[i]
        
        mfv_sum += mfv
        volume_sum += window_volume[i]

    # CMF is the sum of Money Flow Volume divided by the sum of Volume over the period
    if volume_sum != 0:
        cmf = mfv_sum / volume_sum
    else:
        # If there's no volume, CMF is undefined, default to 0.0
        cmf = 0.0
    return cmf

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on the Chaikin Money Flow (CMF) indicator.

    A Buy signal is generated when CMF crosses above a positive threshold (e.g., 0.10).
    A Sell signal is generated when CMF crosses below a negative threshold (e.g., -0.10).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # Ensure we have enough historical data to calculate CMF
        if len(candles) < CMF_PERIOD:
            continue

        # Extract required data points for CMF calculation
        high_prices = [c.high for c in candles]
        low_prices = [c.low for c in candles]
        close_prices = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        timestamps = [c.hour for c in candles]

        cmf_values: list[float] = []

        # Calculate CMF for each possible window, starting from the first full CMF_PERIOD
        for i in range(CMF_PERIOD - 1, len(candles)):
            window_high = high_prices[i - CMF_PERIOD + 1 : i + 1]
            window_low = low_prices[i - CMF_PERIOD + 1 : i + 1]
            window_close = close_prices[i - CMF_PERIOD + 1 : i + 1]
            window_volume = volumes[i - CMF_PERIOD + 1 : i + 1]

            current_cmf = _calculate_cmf_for_window(window_high, window_low, window_close, window_volume)
            cmf_values.append(current_cmf)

            # Generate signals based on CMF crossovers
            # We need at least two CMF values to detect a crossover
            if len(cmf_values) >= 2:
                prev_cmf = cmf_values[-2]
                current_cmf_val = cmf_values[-1] # Renamed to avoid conflict with `current_cmf` from the loop
                current_price = close_prices[i]
                current_timestamp = timestamps[i]

                # Buy signal: CMF crosses above the positive threshold
                if prev_cmf <= BUY_THRESHOLD and current_cmf_val > BUY_THRESHOLD:
                    signals.append(BuySignal(
                        pair=pair,
                        timestamp=current_timestamp,
                        price=current_price,
                        rule_id=RULE_ID
                    ))
                # Sell signal: CMF crosses below the negative threshold
                elif prev_cmf >= SELL_THRESHOLD and current_cmf_val < SELL_THRESHOLD:
                    signals.append(SellSignal(
                        pair=pair,
                        timestamp=current_timestamp,
                        price=current_price,
                        rule_id=RULE_ID
                    ))
    return signals