from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Parameters ---
ADX_PERIOD = 7
ADX_TREND_THRESHOLD = 25
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
EMA_SHORT_PERIOD = 9  # Corresponds to Short_EMA in pseudocode
EMA_MEDIUM_PERIOD = 20  # Corresponds to Medium_EMA in pseudocode (was EMA_LONG_PERIOD in original rule)
EMA_TREND_PERIOD = 50  # NEW: Corresponds to Long_EMA for trend confirmation in pseudocode

# Minimum number of warm candles required for all calculations
# ADX needs 2*ADX_PERIOD + 1
# RSI needs RSI_PERIOD + 1
# EMAs for crossover and trend confirmation:
#   For current EMAs: max(EMA_SHORT_PERIOD, EMA_MEDIUM_PERIOD, EMA_TREND_PERIOD)
#   For previous EMAs (for crossover): max(EMA_SHORT_PERIOD, EMA_MEDIUM_PERIOD) + 1 (since prices[:-1] is used)
#   So, max(max(EMA_SHORT_PERIOD, EMA_MEDIUM_PERIOD) + 1, EMA_TREND_PERIOD)
MIN_CANDLES_REQUIRED = max(
    2 * ADX_PERIOD + 1,
    RSI_PERIOD + 1,
    max(EMA_MEDIUM_PERIOD + 1, EMA_TREND_PERIOD) # Max of periods needed for current/previous EMAs
)

# --- Helper Functions for Indicators ---

def _calculate_ema(prices: np.ndarray, period: int) -> float:
    """Calculates the Exponential Moving Average for the last price in the series."""
    if len(prices) < period:
        return np.nan

    ema = np.zeros_like(prices)
    alpha = 2 / (period + 1)

    # Initial SMA for the first 'period' values
    ema[period - 1] = np.mean(prices[:period])

    # Apply EMA formula for subsequent values
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    
    return ema[-1]

def _calculate_rsi(prices: np.ndarray, period: int) -> float:
    """Calculates the Relative Strength Index for the last price in the series."""
    if len(prices) < period + 1:
        return np.nan

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(gains)
    avg_loss = np.zeros_like(losses)

    # Initial average gain/loss (simple average over 'period' of the diffs)
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    # Wilder's smoothing for subsequent averages
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    if avg_loss[-1] == 0:
        return 100.0 if avg_gain[-1] > 0 else 50.0 # Prevent division by zero, return neutral if no losses
    
    rs = avg_gain[-1] / avg_loss[-1]
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calculate_adx(high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray, period: int) -> float:
    """Calculates the Average Directional Index for the last candle."""
    # Need at least 2*period + 1 candles to calculate the final ADX value
    if len(high_prices) < 2 * period + 1:
        return np.nan

    # Calculate True Range (TR)
    tr_values = np.zeros(len(high_prices))
    for i in range(1, len(high_prices)):
        tr_values[i] = max(high_prices[i] - low_prices[i],
                           abs(high_prices[i] - close_prices[i-1]),
                           abs(low_prices[i] - close_prices[i-1]))

    # Calculate Directional Movement (+DM, -DM)
    plus_dm_values = np.zeros(len(high_prices))
    minus_dm_values = np.zeros(len(high_prices))
    for i in range(1, len(high_prices)):
        up_move = high_prices[i] - high_prices[i-1]
        down_move = low_prices[i-1] - low_prices[i]

        if up_move > down_move and up_move > 0:
            plus_dm_values[i] = up_move
        else:
            plus_dm_values[i] = 0

        if down_move > up_move and down_move > 0:
            minus_dm_values[i] = down_move
        else:
            minus_dm_values[i] = 0

    # Wilder's smoothing function
    def _wilder_smooth(data: np.ndarray, smooth_period: int, start_idx: int = 1) -> np.ndarray:
        smoothed = np.zeros_like(data)
        
        # Ensure enough data for initial sum
        if start_idx + smooth_period > len(data):
            return np.full_like(data, np.nan) 

        # First value is simple sum of 'smooth_period' elements
        smoothed[start_idx + smooth_period - 1] = np.sum(data[start_idx : start_idx + smooth_period])
        for i in range(start_idx + smooth_period, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / smooth_period) + data[i]
        return smoothed

    # Smooth TR, +DM, -DM (start_idx=1 for TR/DM values)
    smoothed_tr = _wilder_smooth(tr_values, period, start_idx=1)
    smoothed_plus_dm = _wilder_smooth(plus_dm_values, period, start_idx=1)
    smoothed_minus_dm = _wilder_smooth(minus_dm_values, period, start_idx=1)
    
    if np.isnan(smoothed_tr[-1]) or np.isnan(smoothed_plus_dm[-1]) or np.isnan(smoothed_minus_dm[-1]):
        return np.nan

    # Calculate DI+ and DI-
    plus_di = np.where(smoothed_tr != 0, (smoothed_plus_dm / smoothed_tr) * 100, 0)
    minus_di = np.where(smoothed_tr != 0, (smoothed_minus_dm / smoothed_tr) * 100, 0)

    # Calculate DX
    dx_values = np.zeros(len(high_prices))
    for i in range(period, len(high_prices)): # DX values become meaningful after the initial smoothing period
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx_values[i] = (abs(plus_di[i] - minus_di[i]) / di_sum) * 100
        else:
            dx_values[i] = 0

    # Calculate ADX (smoothed DX). ADX itself is a smoothed version of DX.
    # The first ADX value is the simple average of the first 'period' DX values (from index `period` to `2*period - 1`).
    # Subsequent ADX values are Wilder's smoothed.
    
    adx_values = np.zeros_like(dx_values)
    
    # Check if there are enough DX values to calculate the initial ADX sum
    if 2 * period > len(dx_values):
        return np.nan
        
    adx_values[2 * period - 1] = np.mean(dx_values[period : 2 * period])

    # Apply Wilder's smoothing for subsequent ADX values
    for i in range(2 * period, len(high_prices)):
        adx_values[i] = (adx_values[i-1] * (period - 1) + dx_values[i]) / period

    return adx_values[-1]


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    RULE_ID = "3f2233fe-091f-4887-896b-7ac3c42deec1"

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Ensure candles are sorted by time (oldest first)
        warm_candles.sort(key=lambda c: c.hour)

        # Extract necessary price arrays for calculations
        high_prices = np.array([c.high for c in warm_candles], dtype=float)
        low_prices = np.array([c.low for c in warm_candles], dtype=float)
        close_prices = np.array([c.close for c in warm_candles], dtype=float)

        # --- Calculate Indicators for current candle ---
        adx = _calculate_adx(high_prices, low_prices, close_prices, ADX_PERIOD)
        ema_short_current = _calculate_ema(close_prices, EMA_SHORT_PERIOD)
        ema_medium_current = _calculate_ema(close_prices, EMA_MEDIUM_PERIOD)
        ema_trend_current = _calculate_ema(close_prices, EMA_TREND_PERIOD) # New long-term EMA for trend confirmation
        rsi = _calculate_rsi(close_prices, RSI_PERIOD)

        # Check for NaN results from insufficient data within indicator functions
        if np.isnan(adx) or np.isnan(ema_short_current) or np.isnan(ema_medium_current) or \
           np.isnan(ema_trend_current) or np.isnan(rsi):
            continue

        current_price = warm_candles[-1].close
        current_timestamp = warm_candles[-1].hour # Using candle close time as signal time

        # --- Determine Market Regime ---
        regime = "RANGING"
        if adx > ADX_TREND_THRESHOLD:
            regime = "TRENDING"

        # --- Generate Signal based on Regime ---
        if regime == "TRENDING":
            # Trend-following: EMA Crossovers, confirmed by Long_EMA direction
            
            # To detect a crossover, we need current and previous values.
            # We already have current values. For previous, we calculate EMAs for the series ending one candle ago.
            # Check if there's enough data to calculate previous EMAs (len(close_prices[:-1]) >= period)
            if len(close_prices) > max(EMA_SHORT_PERIOD, EMA_MEDIUM_PERIOD):
                ema_short_prev = _calculate_ema(close_prices[:-1], EMA_SHORT_PERIOD)
                ema_medium_prev = _calculate_ema(close_prices[:-1], EMA_MEDIUM_PERIOD)

                if not np.isnan(ema_short_prev) and not np.isnan(ema_medium_prev):
                    # Buy signal: Short_EMA crosses above Medium_EMA AND current_price > Long_EMA
                    if ema_short_prev < ema_medium_prev and ema_short_current > ema_medium_current:
                        if current_price > ema_trend_current: # Confirm uptrend direction
                            signals.append(BuySignal(
                                pair=pair,
                                timestamp=current_timestamp,
                                price=current_price,
                                rule_id=RULE_ID,
                                confidence=min(1.0, adx/100) # ADX strength as confidence, capped at 1.0
                            ))
                    # Sell signal: Short_EMA crosses below Medium_EMA AND current_price < Long_EMA
                    elif ema_short_prev > ema_medium_prev and ema_short_current < ema_medium_current:
                        if current_price < ema_trend_current: # Confirm downtrend direction
                            signals.append(SellSignal(
                                pair=pair,
                                timestamp=current_timestamp,
                                price=current_price,
                                rule_id=RULE_ID,
                                confidence=min(1.0, adx/100) # ADX strength as confidence, capped at 1.0
                            ))
        elif regime == "RANGING":
            # Mean-reversion: RSI signals
            if rsi < RSI_OVERSOLD:
                signals.append(BuySignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=min(1.0, (RSI_OVERSOLD - rsi) / RSI_OVERSOLD) # Deeper oversold = higher confidence
                ))
            elif rsi > RSI_OVERBOUGHT:
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=min(1.0, (rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)) # Higher overbought = higher confidence
                ))
    return signals