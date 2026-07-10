"""Rule 03 — Volatility-Adaptive Bollinger Band Deviation."""
from __future__ import annotations

import numpy as np

from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Constants ---
N_PERIODS_BB = 20  # Periods for Bollinger Band SMA and StdDev
M_PERIODS_ATR = 14  # Periods for Average True Range

K_MIN = 1.0  # Minimum K multiplier (e.g., during very low volatility)
K_MAX = 3.0  # Maximum K multiplier (e.g., during very high volatility)

# Thresholds for normalizing ATR to determine K_dynamic.
# These represent normalized ATR (ATR / SMA) values.
# If normalized_ATR <= ATR_NORM_MIN_THRESHOLD, K_dynamic will be K_MIN.
# If normalized_ATR >= ATR_NORM_MAX_THRESHOLD, K_dynamic will be K_MAX.
# In between, K_dynamic will be linearly interpolated.
ATR_NORM_MIN_THRESHOLD = 0.005  # Example: 0.5% average true range relative to price
ATR_NORM_MAX_THRESHOLD = 0.02  # Example: 2% average true range relative to price

# Minimum number of warm candles required
# N_PERIODS_BB for SMA/StdDev
# M_PERIODS_ATR + 1 for ATR (to get previous close for the first TR in the window)
MIN_CANDLES = max(N_PERIODS_BB, M_PERIODS_ATR + 1)


def _calculate_atr(candles: list[WarmCandle], period: int) -> float | None:
    """
    Calculates the Average True Range (ATR) for a given list of candles.
    Requires `period + 1` candles to calculate `period` True Ranges.
    The input `candles` list should be ordered from oldest to newest.
    """
    if len(candles) < period + 1:
        return None

    true_ranges_values = []
    # Iterate from the (period+1)-th oldest candle to the newest candle.
    # The list `candles` should contain at least `period + 1` elements.
    # We calculate `period` TRs, using `candles[i]` and `candles[i-1]`.
    # `candles[0]` will be used as the `prev_close` for `candles[1]`.
    for i in range(1, period + 1):
        current_candle = candles[i]
        prev_candle = candles[i - 1]

        tr = max(
            current_candle.high - current_candle.low,
            abs(current_candle.high - prev_candle.close),
            abs(current_candle.low - prev_candle.close),
        )
        true_ranges_values.append(tr)

    if not true_ranges_values:  # Should not happen if len(candles) >= period + 1
        return None

    return float(np.mean(true_ranges_values))


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm data for both BB and ATR calculations
        if len(pair_data.warm) < MIN_CANDLES:
            continue

        # 1. Calculate SMA and StdDev over N_PERIODS_BB
        # We need the most recent N_PERIODS_BB candles for BB calculation
        bb_candles_for_stats = pair_data.warm[-N_PERIODS_BB:]
        closes = np.array([c.close for c in bb_candles_for_stats])

        # Defensive check, though MIN_CANDLES should cover this.
        if len(closes) < N_PERIODS_BB:
            continue

        mean = float(np.mean(closes))
        std = float(np.std(closes, ddof=1))  # Use sample standard deviation

        if std == 0:
            # If standard deviation is zero, all closes are the same. Bands collapse.
            # This makes the indicator meaningless, so skip.
            continue

        # 2. Calculate Average True Range (ATR) over M_PERIODS_ATR
        # We need M_PERIODS_ATR + 1 candles for _calculate_atr
        atr_candles_for_calculation = pair_data.warm[-(M_PERIODS_ATR + 1) :]

        atr = _calculate_atr(atr_candles_for_calculation, M_PERIODS_ATR)

        # ATR can be None if insufficient candles, or 0 if no price movement.
        if atr is None or atr == 0:
            continue

        # 3. Define a dynamic multiplier K_dynamic = f(ATR)
        # Normalize ATR by the SMA of the BB closes to get a volatility percentage
        normalized_atr = atr / mean

        # Linear interpolation for K_dynamic based on normalized_atr
        # K_dynamic will be clamped between K_MIN and K_MAX

        if ATR_NORM_MAX_THRESHOLD == ATR_NORM_MIN_THRESHOLD:
            # Avoid division by zero if thresholds are identical
            k_dynamic = (K_MIN + K_MAX) / 2
        else:
            # Calculate interpolation factor: 0 if normalized_atr is at or below min threshold,
            # 1 if at or above max threshold, and linearly interpolated in between.
            interpolation_factor = (normalized_atr - ATR_NORM_MIN_THRESHOLD) / (
                ATR_NORM_MAX_THRESHOLD - ATR_NORM_MIN_THRESHOLD
            )

            # Clamp the factor to ensure K_dynamic stays within K_MIN and K_MAX
            interpolation_factor = max(0.0, min(1.0, interpolation_factor))

            k_dynamic = K_MIN + (K_MAX - K_MIN) * interpolation_factor

        # 4. Calculate Upper Band = SMA + K_dynamic * StdDev
        # 5. Calculate Lower Band = SMA - K_dynamic * StdDev
        upper_band = mean + k_dynamic * std
        lower_band = mean - k_dynamic * std

        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # 6. Generate Buy signal if current price < Lower Band.
        # 7. Generate Sell signal if current price > Upper Band.
        if current_price < lower_band:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif current_price > upper_band:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals