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

ADX_PERIOD = 14  # Periods for Average Directional Index (ADX) calculation
ADX_THRESHOLD = 25  # ADX value below which a market is considered non-trending

# Minimum number of warm candles required
# N_PERIODS_BB for SMA/StdDev
# M_PERIODS_ATR + 1 for ATR (to get previous close for the first TR in the window)
# 3 * ADX_PERIOD - 1 for a robust ADX calculation (Wilder's smoothing)
MIN_CANDLES = max(N_PERIODS_BB, M_PERIODS_ATR + 1, (3 * ADX_PERIOD - 1))


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


def _calculate_adx(candles: list[WarmCandle], period: int) -> float | None:
    """
    Calculates the Average Directional Index (ADX) for a given list of candles
    using Wilder's smoothing method.
    Requires at least (3 * period - 1) candles for a robust calculation.
    The input `candles` list should be ordered from oldest to newest.
    """
    if len(candles) < (3 * period - 1):
        return None

    trs = []
    plus_dms = []
    minus_dms = []

    # Calculate True Range (TR), Positive Directional Movement (+DM), and Negative Directional Movement (-DM)
    # for each candle starting from the second one.
    # We need `len(candles) - 1` TR/DM values.
    for i in range(1, len(candles)):
        current_candle = candles[i]
        prev_candle = candles[i - 1]

        # True Range
        tr = max(
            current_candle.high - current_candle.low,
            abs(current_candle.high - prev_candle.close),
            abs(current_candle.low - prev_candle.close),
        )
        trs.append(tr)

        # Directional Movement
        up_move = current_candle.high - prev_candle.high
        down_move = prev_candle.low - current_candle.low

        plus_dm = 0.0
        minus_dm = 0.0

        if up_move > down_move and up_move > 0:
            plus_dm = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm = down_move
        
        plus_dms.append(plus_dm)
        minus_dms.append(minus_dm)

    # Check if we have enough TR/DM values for initial smoothing
    if len(trs) < period:
        return None

    # Wilder's Smoothing for TR, +DM, -DM
    # Initial values are simple sums over the first `period` TR/DM values.
    # Subsequent values use Wilder's smoothing formula.
    smoothed_plus_dms = [sum(plus_dms[:period])]
    smoothed_minus_dms = [sum(minus_dms[:period])]
    smoothed_trs = [sum(trs[:period])]

    for i in range(period, len(plus_dms)):
        smoothed_plus_dms.append(
            (smoothed_plus_dms[-1] * (period - 1) + plus_dms[i]) / period
        )
        smoothed_minus_dms.append(
            (smoothed_minus_dms[-1] * (period - 1) + minus_dms[i]) / period
        )
        smoothed_trs.append(
            (smoothed_trs[-1] * (period - 1) + trs[i]) / period
        )

    # Calculate DI+ and DI-
    plus_dis = []
    minus_dis = []
    for i in range(len(smoothed_trs)):
        if smoothed_trs[i] == 0:
            plus_dis.append(0.0)
            minus_dis.append(0.0)
        else:
            plus_dis.append(100 * smoothed_plus_dms[i] / smoothed_trs[i])
            minus_dis.append(100 * smoothed_minus_dms[i] / smoothed_trs[i])

    # Calculate DX (Directional Index)
    dxs = []
    for i in range(len(plus_dis)):
        di_sum = plus_dis[i] + minus_dis[i]
        if di_sum == 0:
            dxs.append(0.0)
        else:
            dxs.append(100 * abs(plus_dis[i] - minus_dis[i]) / di_sum)

    # Check if we have enough DX values for ADX smoothing
    if len(dxs) < period:
        return None

    # Wilder's Smoothing for ADX
    # Initial ADX is the average of the first `period` DX values.
    # Subsequent values use Wilder's smoothing formula.
    adx_values = [sum(dxs[:period]) / period]

    for i in range(period, len(dxs)):
        adx_values.append(
            (adx_values[-1] * (period - 1) + dxs[i]) / period
        )
    
    return adx_values[-1]  # Return the latest ADX value


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm data for BB, ATR, and ADX calculations
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

        # 6. Calculate ADX
        # We need `3 * ADX_PERIOD - 1` candles for _calculate_adx
        adx_candles_for_calculation = pair_data.warm[-(3 * ADX_PERIOD - 1):]
        adx = _calculate_adx(adx_candles_for_calculation, ADX_PERIOD)

        # ADX can be None if insufficient candles, or if intermediate calculations result in zero.
        if adx is None:
            continue

        # 7. Apply ADX trend filter: Only generate signals if ADX is below the threshold
        if adx < ADX_THRESHOLD:
            # 8. Generate Buy signal if current price < Lower Band.
            # 9. Generate Sell signal if current price > Upper Band.
            if current_price < lower_band:
                signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
            elif current_price > upper_band:
                signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals