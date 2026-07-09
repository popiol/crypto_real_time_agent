from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Parameters
KUMO_PERIOD_TENKAN = 9
KUMO_PERIOD_KIJUN = 26
KUMO_PERIOD_SENKOU_SPAN_B = 52
KUMO_PERIOD_CHIKOU = 26
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 50
RSI_SELL_THRESHOLD = 50
PRICE_BREAKOUT_CONFIRMATION_CANDLES = 2

# Minimum number of candles required for calculations
# For Senkou Span A/B at current time 't', we need raw values from 't - KUMO_PERIOD_KIJUN'.
# The raw Senkou Span B at 't - KUMO_PERIOD_KIJUN' needs data back KUMO_PERIOD_SENKOU_SPAN_B periods.
# Thus, the earliest data point needed for Senkou Spans relevant to current time is
# (current_idx - KUMO_PERIOD_KIJUN) - KUMO_PERIOD_SENKOU_SPAN_B + 1.
# This means we need at least KUMO_PERIOD_KIJUN + KUMO_PERIOD_SENKOU_SPAN_B candles for the Senkou Spans
# to be available at the current time's index (len(candles) - 1).
MIN_CANDLES_ICHIMOKU_CORE = KUMO_PERIOD_KIJUN + KUMO_PERIOD_SENKOU_SPAN_B # 26 + 52 = 78

# For Chikou Span, we need KUMO_PERIOD_CHIKOU + 1 candles to get the value for the latest candle.
MIN_CANDLES_CHIKOU = KUMO_PERIOD_CHIKOU + 1 # 26 + 1 = 27

# For RSI, we need RSI_PERIOD + 1 candles for the latest candle.
MIN_CANDLES_RSI = RSI_PERIOD + 1 # 14 + 1 = 15

# The overall minimum required candles must cover the largest lookback for the current candle's indicators.
# Additionally, for breakout confirmation, we need to check `PRICE_BREAKOUT_CONFIRMATION_CANDLES` candles.
# If PRICE_BREAKOUT_CONFIRMATION_CANDLES is N, we check indices `current_idx`, `current_idx-1`, ..., `current_idx - N + 1`.
# All these N candles must have valid Ichimoku and RSI values.
# So, the earliest index to have valid indicators is `current_idx - N + 1`.
# This earliest index must satisfy `max(MIN_CANDLES_ICHIMOKU_CORE, MIN_CANDLES_CHIKOU, MIN_CANDLES_RSI) - 1`.
# Therefore, `(len(candles) - 1) - (PRICE_BREAKOUT_CONFIRMATION_CANDLES - 1)` must be `>= max_lookback_idx`.
# `len(candles) - PRICE_BREAKOUT_CONFIRMATION_CANDLES >= max_lookback_idx`.
# `len(candles) >= max_lookback_idx + PRICE_BREAKOUT_CONFIRMATION_CANDLES`.
# `max_lookback_idx` is `max(MIN_CANDLES_ICHIMOKU_CORE, MIN_CANDLES_CHIKOU, MIN_CANDLES_RSI) - 1`.
MIN_CANDLES_FOR_RULE = max(MIN_CANDLES_ICHIMOKU_CORE, MIN_CANDLES_CHIKOU, MIN_CANDLES_RSI) + PRICE_BREAKOUT_CONFIRMATION_CANDLES - 1


def _calculate_ichimoku(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates Ichimoku components."""

    # Tenkan-sen (Conversion Line)
    tenkan_sen = (
        np.array([np.max(highs[max(0, i - KUMO_PERIOD_TENKAN + 1) : i + 1]) for i in range(len(highs))])
        + np.array([np.min(lows[max(0, i - KUMO_PERIOD_TENKAN + 1) : i + 1]) for i in range(len(lows))])
    ) / 2

    # Kijun-sen (Base Line)
    kijun_sen = (
        np.array([np.max(highs[max(0, i - KUMO_PERIOD_KIJUN + 1) : i + 1]) for i in range(len(highs))])
        + np.array([np.min(lows[max(0, i - KUMO_PERIOD_KIJUN + 1) : i + 1]) for i in range(len(lows))])
    ) / 2

    # Senkou Span A (Leading Span A, plotted KUMO_PERIOD_KIJUN periods ahead)
    senkou_span_a_raw = (tenkan_sen + kijun_sen) / 2
    senkou_span_a = np.full_like(senkou_span_a_raw, np.nan)
    # Shift raw values back to align with current time for plotting ahead
    senkou_span_a[KUMO_PERIOD_KIJUN:] = senkou_span_a_raw[:-KUMO_PERIOD_KIJUN]

    # Senkou Span B (Leading Span B, plotted KUMO_PERIOD_KIJUN periods ahead)
    senkou_span_b_raw = (
        np.array([np.max(highs[max(0, i - KUMO_PERIOD_SENKOU_SPAN_B + 1) : i + 1]) for i in range(len(highs))])
        + np.array([np.min(lows[max(0, i - KUMO_PERIOD_SENKOU_SPAN_B + 1) : i + 1]) for i in range(len(lows))])
    ) / 2
    senkou_span_b = np.full_like(senkou_span_b_raw, np.nan)
    # Shift raw values back to align with current time for plotting ahead
    senkou_span_b[KUMO_PERIOD_KIJUN:] = senkou_span_b_raw[:-KUMO_PERIOD_KIJUN]

    # Kumo (Cloud) boundaries
    kumo_upper = np.maximum(senkou_span_a, senkou_span_b)
    kumo_lower = np.minimum(senkou_span_a, senkou_span_b)

    # Chikou Span (Lagging Span, current closing price shifted KUMO_PERIOD_CHIKOU periods back)
    chikou_span = np.full_like(closes, np.nan)
    chikou_span[KUMO_PERIOD_CHIKOU:] = closes[:-KUMO_PERIOD_CHIKOU]

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, kumo_upper, kumo_lower, chikou_span


def _calculate_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Calculates the Relative Strength Index (RSI)."""
    if len(closes) <= period:
        return np.full_like(closes, np.nan)

    diff = np.diff(closes)
    up = np.maximum(0, diff)
    down = np.maximum(0, -diff)

    avg_gain = np.zeros_like(closes)
    avg_loss = np.zeros_like(closes)
    rsi = np.full_like(closes, np.nan)

    # Initial SMA for the first 'period' values
    avg_gain[period] = np.mean(up[:period])
    avg_loss[period] = np.mean(down[:period])

    # Wilder's smoothing
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + up[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + down[i - 1]) / period

    # Calculate RS and RSI
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    # Handle edge cases where avg_loss or avg_gain is zero
    rsi[period:][avg_loss[period:] == 0] = 100 # Only gains, no losses
    rsi[period:][avg_gain[period:] == 0] = 0   # Only losses, no gains

    return rsi


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm
        if len(candles) < MIN_CANDLES_FOR_RULE:
            continue

        # Extract necessary data for calculations
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])

        # Calculate Ichimoku components
        _, _, _, _, kumo_upper, kumo_lower, chikou_span = _calculate_ichimoku(highs, lows, closes)

        # Calculate RSI
        rsi = _calculate_rsi(closes, RSI_PERIOD)

        # Determine the current index (latest candle)
        current_idx = len(candles) - 1

        # Check if all required indicators are valid for the current candle
        if (
            np.isnan(kumo_upper[current_idx])
            or np.isnan(kumo_lower[current_idx])
            or np.isnan(chikou_span[current_idx])
            or np.isnan(rsi[current_idx])
        ):
            continue

        current_close = closes[current_idx]
        current_kumo_upper = kumo_upper[current_idx]
        current_kumo_lower = kumo_lower[current_idx]
        current_chikou_span = chikou_span[current_idx]
        current_rsi = rsi[current_idx]
        current_timestamp = candles[current_idx].hour
        current_price = candles[current_idx].close

        # Helper functions for breakout confirmation
        def _all_candles_close_above_kumo_upper(num_candles: int) -> bool:
            # Check the last `num_candles` (including current_idx)
            start_idx = current_idx - num_candles + 1
            if start_idx < 0:
                return False # Not enough history for the confirmation period
            
            for i in range(start_idx, current_idx + 1):
                if closes[i] <= kumo_upper[i] or np.isnan(kumo_upper[i]):
                    return False
            return True

        def _all_candles_close_below_kumo_lower(num_candles: int) -> bool:
            # Check the last `num_candles` (including current_idx)
            start_idx = current_idx - num_candles + 1
            if start_idx < 0:
                return False # Not enough history for the confirmation period
            
            for i in range(start_idx, current_idx + 1):
                if closes[i] >= kumo_lower[i] or np.isnan(kumo_lower[i]):
                    return False
            return True

        # Buy Signal Logic
        # Price must be above Kumo_Upper_Boundary
        # Previous candle must have been inside or below Kumo
        # Confirmed breakout (last PRICE_BREAKOUT_CONFIRMATION_CANDLES close above Kumo_Upper_Boundary)
        # Chikou Span must be above current price (bullish momentum)
        # RSI must be above RSI_BUY_THRESHOLD
        if (
            current_close > current_kumo_upper
            and closes[current_idx - 1] <= kumo_upper[current_idx - 1] # Previous candle was inside or below Kumo
            and _all_candles_close_above_kumo_upper(PRICE_BREAKOUT_CONFIRMATION_CANDLES)
            and current_chikou_span > current_close
            and current_rsi > RSI_BUY_THRESHOLD
        ):
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_price,
                    rule_id="0406dabd-17cf-4318-83d1-0a8e9e512fce",
                    confidence=None # Or a calculated confidence if desired
                )
            )

        # Sell Signal Logic
        # Price must be below Kumo_Lower_Boundary
        # Previous candle must have been inside or above Kumo
        # Confirmed breakout (last PRICE_BREAKOUT_CONFIRMATION_CANDLES close below Kumo_Lower_Boundary)
        # Chikou Span must be below current price (bearish momentum)
        # RSI must be below RSI_SELL_THRESHOLD
        elif (
            current_close < current_kumo_lower
            and closes[current_idx - 1] >= kumo_lower[current_idx - 1] # Previous candle was inside or above Kumo
            and _all_candles_close_below_kumo_lower(PRICE_BREAKOUT_CONFIRMATION_CANDLES)
            and current_chikou_span < current_close
            and current_rsi < RSI_SELL_THRESHOLD
        ):
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_price,
                    rule_id="0406dabd-17cf-4318-83d1-0a8e9e512fce",
                    confidence=None # Or a calculated confidence if desired
                )
            )

    return signals