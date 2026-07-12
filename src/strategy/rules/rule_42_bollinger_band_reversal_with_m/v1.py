from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule Parameters ---
# These are defined as constants based on the pseudocode examples.
PERIOD_BB = 20
STD_DEV_BB = 2
PERIOD_MFI = 14
OVERSOLD_MFI_THRESHOLD = 20
OVERBOUGHT_MFI_THRESHOLD = 80
PERIOD_VOLUME_SMA = 20
VOLUME_MULTIPLIER = 1.5
PERIOD_SHORT_EMA = 10
PERIOD_LONG_EMA = 50
TREND_SLOPE_PERIOD = 5

# --- Helper Functions for Indicators ---

def _sma(data: np.ndarray, window: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < window:
        return np.array([])
    return np.convolve(data, np.ones(window)/window, mode='valid')

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Exponential Moving Average."""
    if len(data) == 0:
        return np.array([])
    alpha = 2 / (period + 1)
    ema_values = np.zeros_like(data)
    ema_values[0] = data[0] # Initialize with the first data point
    for i in range(1, len(data)):
        ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
    return ema_values

def _mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Calculates Money Flow Index (MFI)."""
    if not (len(high) == len(low) == len(close) == len(volume)) or len(close) < period + 1:
        return np.array([])

    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_mf = np.zeros_like(money_flow)
    negative_mf = np.zeros_like(money_flow)

    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_mf[i] = money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_mf[i] = money_flow[i]

    # Rolling sum for PMF and NMF over the given period
    # The 'valid' mode means the convolution output is of length len(data) - period + 1
    pmf_sum = np.convolve(positive_mf, np.ones(period), 'valid')
    nmf_sum = np.convolve(negative_mf, np.ones(period), 'valid')

    # Handle division by zero for nmf_sum by setting money_ratio to infinity
    # when nmf_sum is zero (implies all money flow is positive).
    money_ratio = np.divide(pmf_sum, nmf_sum, out=np.full_like(pmf_sum, np.inf), where=nmf_sum!=0)

    mfi_values = 100 - (100 / (1 + money_ratio))
    return mfi_values

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Bollinger Band Reversal with MFI, Volume, and Trend Confirmation rule.
    Generates Buy/Sell signals based on combined conditions.
    """
    signals: list[BuySignal | SellSignal] = []
    rule_id = "32a93297-06e1-4424-9311-85149f50abfd"

    # Minimum candles required for all indicators:
    # - Bollinger Bands: PERIOD_BB candles for current calculation.
    # - MFI: PERIOD_MFI + 1 candles for the first MFI value, and 2 values for comparison (so PERIOD_MFI + 2 total).
    # - Volume SMA: PERIOD_VOLUME_SMA candles.
    # - EMAs: PERIOD_LONG_EMA candles for the longest EMA to stabilize.
    # - EMA Slope: PERIOD_LONG_EMA + TREND_SLOPE_PERIOD candles to get reliable slope.
    min_candles_required = max(
        PERIOD_BB,
        PERIOD_MFI + 2,
        PERIOD_VOLUME_SMA,
        PERIOD_LONG_EMA,
        PERIOD_LONG_EMA + TREND_SLOPE_PERIOD
    )

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # Ensure enough data is available
        if len(candles) < min_candles_required:
            continue

        # Extract data into numpy arrays. WarmCandle list is assumed to be ordered oldest to newest.
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        timestamps = [c.hour for c in candles]

        # --- Calculate Indicators ---

        # 1. Bollinger Bands (calculated based on the last PERIOD_BB candles)
        bb_closes_window = closes[-PERIOD_BB:]
        bb_middle = np.mean(bb_closes_window)
        bb_std = np.std(bb_closes_window)
        bb_upper = bb_middle + STD_DEV_BB * bb_std
        bb_lower = bb_middle - STD_DEV_BB * bb_std

        current_close = closes[-1]

        # 2. Money Flow Index (MFI)
        mfi_values = _mfi(highs, lows, closes, volumes, PERIOD_MFI)
        if len(mfi_values) < 2: # Need at least 2 MFI values for current vs previous comparison
            continue
        current_mfi = mfi_values[-1]
        previous_mfi = mfi_values[-2]

        # 3. Volume SMA
        volume_sma_values = _sma(volumes, PERIOD_VOLUME_SMA)
        if len(volume_sma_values) == 0: # This check should ideally be covered by min_candles_required
            continue
        current_volume_sma = volume_sma_values[-1]
        current_volume = volumes[-1]

        # 4. Exponential Moving Averages (EMAs)
        ema_short_series = _ema(closes, PERIOD_SHORT_EMA)
        ema_long_series = _ema(closes, PERIOD_LONG_EMA)

        # Check if EMA series are long enough for slope calculation
        if len(ema_short_series) < TREND_SLOPE_PERIOD or len(ema_long_series) < TREND_SLOPE_PERIOD:
            continue

        current_ema_short = ema_short_series[-1]
        current_ema_long = ema_long_series[-1]

        # 5. EMA Slopes
        # Slope is calculated as (current EMA - EMA_N_bars_ago) / (N-1).
        # If TREND_SLOPE_PERIOD is 1, slope is considered 0.
        ema_short_slope = (current_ema_short - ema_short_series[-TREND_SLOPE_PERIOD]) / (TREND_SLOPE_PERIOD - 1) if TREND_SLOPE_PERIOD > 1 else 0.0
        ema_long_slope = (current_ema_long - ema_long_series[-TREND_SLOPE_PERIOD]) / (TREND_SLOPE_PERIOD - 1) if TREND_SLOPE_PERIOD > 1 else 0.0

        # --- Buy Signal Conditions ---
        buy_cond1 = current_close < bb_lower
        buy_cond2 = current_mfi < OVERSOLD_MFI_THRESHOLD
        buy_cond3 = current_mfi > previous_mfi # MFI is increasing
        buy_cond4 = current_volume > current_volume_sma * VOLUME_MULTIPLIER
        buy_cond5 = (current_ema_short > current_ema_long and ema_short_slope > 0) or (ema_long_slope > 0)

        if buy_cond1 and buy_cond2 and buy_cond3 and buy_cond4 and buy_cond5:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamps[-1],
                price=current_close,
                rule_id=rule_id
            ))

        # --- Sell Signal Conditions ---
        sell_cond1 = current_close > bb_upper
        sell_cond2 = current_mfi > OVERBOUGHT_MFI_THRESHOLD
        sell_cond3 = current_mfi < previous_mfi # MFI is decreasing
        sell_cond4 = current_volume > current_volume_sma * VOLUME_MULTIPLIER
        sell_cond5 = (current_ema_short < current_ema_long and ema_short_slope < 0) or (ema_long_slope < 0)

        if sell_cond1 and sell_cond2 and sell_cond3 and sell_cond4 and sell_cond5:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamps[-1],
                price=current_close,
                rule_id=rule_id
            ))

    return signals