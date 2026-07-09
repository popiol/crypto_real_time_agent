"""Rule 266701ac-c78c-48ec-8000-2075c06d90e5 — Keltner Channel Breakout with ADX and Volume Confirmation (v2)."""
from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle
from datetime import datetime

# --- IMPORTANT ASSUMPTION ---
# This rule relies on the 'WarmCandle' model having a 'volume' attribute representing
# the trading volume for the hour. The provided 'WarmCandle' definition does NOT
# include volume. For this rule to function correctly, the 'WarmCandle' model
# must be extended to include 'volume: float'.
# If 'volume' is not present, accessing 'c.volume' will result in an AttributeError.
# If 'volume' is present but always 0, the volume confirmation will always be false.
# ----------------------------

# Parameters
Keltner_Period = 20 # Used for EMA and ATR calculation periods
ATR_Multiplier = 2.0
ADX_Period = 14
ADX_Threshold = 25
Volume_MA_Period = 20
Volume_Multiplier = 1.5

# Minimum number of candles required for calculations:
#   - EMA(Keltner_Period): needs `Keltner_Period` candles for the first value (at index Keltner_Period-1).
#   - ATR(Keltner_Period): needs 1 for prev_close, then `Keltner_Period` TRs for initial average (at index Keltner_Period).
#                     So, `Keltner_Period + 1` candles.
#   - ADX(ADX_Period): needs `2 * ADX_Period` candles for the first ADX value (at index 2*ADX_Period-1).
#   - SMA(Volume_MA_Period): needs `Volume_MA_Period` candles for the first value (at index Volume_MA_Period-1).
MIN_CANDLES = max(Keltner_Period, Keltner_Period + 1, 2 * ADX_Period, Volume_MA_Period)

def _calculate_ema_full(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates Exponential Moving Average (EMA) returning full array with NaNs for initial non-calculable points."""
    ema = np.full_like(prices, np.nan, dtype=float)
    if len(prices) < period:
        return ema
    
    # The first EMA value is often initialized as the Simple Moving Average (SMA) of the first 'period' values.
    # This value is placed at index `period - 1`.
    ema[period - 1] = np.mean(prices[:period])
    alpha = 2 / (period + 1)
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def _calculate_atr_full(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculates Average True Range (ATR) returning full array with NaNs for initial non-calculable points."""
    atr = np.full_like(high, np.nan, dtype=float)
    if len(high) < 2: # Need at least two candles to calculate first True Range (TR)
        return atr
    
    tr = np.full_like(high, np.nan, dtype=float)
    for i in range(1, len(high)):
        # True Range: max of (H-L, |H-PrevC|, |L-PrevC|)
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # We need `period` valid TR values to calculate the initial ATR (SMA of TRs).
    # The first valid TR is at index 1. So, we need `tr[1]` to `tr[period]`.
    if len(tr[1:]) < period: # Check if there are 'period' valid TRs
        return atr
        
    # Initial ATR is SMA of the first 'period' True Ranges.
    # This value is placed at index `period` of the `atr` array.
    atr[period] = np.mean(tr[1:period+1])
    
    # Subsequent ATR values use Wilder's smoothing method.
    for i in range(period + 1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr

def _calculate_adx_full(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates ADX, Plus_DI, Minus_DI returning full arrays with NaNs for initial non-calculable points."""
    adx = np.full_like(high, np.nan, dtype=float)
    plus_di = np.full_like(high, np.nan, dtype=float)
    minus_di = np.full_like(high, np.nan, dtype=float)

    if len(high) < 2 * period: # Minimum candles for ADX calculation
        return adx, plus_di, minus_di

    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)

    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]

        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    # Wilder's smoothing for DM and TR.
    # The first smoothed value appears at index `period`.
    smoothed_plus_dm = np.full_like(high, np.nan, dtype=float)
    smoothed_minus_dm = np.full_like(high, np.nan, dtype=float)
    smoothed_tr = np.full_like(high, np.nan, dtype=float)

    # Initial smoothed value (at index `period`) is the sum of the first `period` valid values (from index 1 to `period`).
    smoothed_plus_dm[period] = np.sum(plus_dm[1:period+1])
    smoothed_minus_dm[period] = np.sum(minus_dm[1:period+1])
    smoothed_tr[period] = np.sum(tr[1:period+1])

    for i in range(period + 1, len(high)):
        smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1] / period) + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1] / period) + minus_dm[i]
        smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1] / period) + tr[i]

    # Calculate DI+ and DI-
    for i in range(period, len(high)): # DI values start from index `period`
        if smoothed_tr[i] != 0:
            plus_di[i] = 100 * (smoothed_plus_dm[i] / smoothed_tr[i])
            minus_di[i] = 100 * (smoothed_minus_dm[i] / smoothed_tr[i])
        else:
            plus_di[i] = 0
            minus_di[i] = 0

    # Calculate DX
    dx = np.full_like(high, np.nan, dtype=float)
    for i in range(period, len(high)): # DX values start from index `period`
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * (abs(plus_di[i] - minus_di[i]) / di_sum)
        else:
            dx[i] = 0

    # Calculate ADX (smoothed DX).
    # The first ADX value (at index `2 * period - 1`) is the SMA of DX for `period` values,
    # specifically `dx[period]` through `dx[2 * period - 1]`.
    if len(dx[period:]) < period: # Check if enough DX values exist to calculate initial ADX
        return adx, plus_di, minus_di
        
    adx[2 * period - 1] = np.mean(dx[period : 2 * period]) 

    # For subsequent ADX values, use Wilder's smoothing.
    for i in range(2 * period, len(high)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di

def _calculate_sma_full(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average (SMA) returning full array with NaNs for initial non-calculable points."""
    sma = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return sma
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i - period + 1 : i + 1])
    return sma

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        if len(warm_candles) < MIN_CANDLES:
            continue

        # Extract numpy arrays for calculations
        high_prices = np.array([c.high for c in warm_candles])
        low_prices = np.array([c.low for c in warm_candles])
        close_prices = np.array([c.close for c in warm_candles])
        
        # --- CRITICAL ASSUMPTION ---
        # This line assumes that each WarmCandle object has a 'volume' attribute.
        # If the WarmCandle model as defined in the project does not include 'volume',
        # this will cause an AttributeError.
        # The 'volume' attribute is expected to be the trading volume for that specific candle's period (e.g., hourly volume).
        try:
            volumes = np.array([c.volume for c in warm_candles])
        except AttributeError:
            # If volume is not available, the rule cannot be fully executed.
            # Print a warning and skip this pair.
            print(f"Warning: WarmCandle for pair {pair} does not have a 'volume' attribute. Skipping rule for this pair.")
            continue
        # ---------------------------

        # Calculate all required indicators
        ema_values = _calculate_ema_full(close_prices, Keltner_Period)
        atr_values = _calculate_atr_full(high_prices, low_prices, close_prices, Keltner_Period)
        adx_values, plus_di_values, minus_di_values = _calculate_adx_full(high_prices, low_prices, close_prices, ADX_Period)
        volume_sma_values = _calculate_sma_full(volumes, Volume_MA_Period)

        # Check if the latest values for all indicators are valid (not NaN).
        # This implicitly handles cases where a helper function returned an array full of NaNs due to insufficient data.
        if np.isnan(ema_values[-1]) or \
           np.isnan(atr_values[-1]) or \
           np.isnan(adx_values[-1]) or \
           np.isnan(plus_di_values[-1]) or \
           np.isnan(minus_di_values[-1]) or \
           np.isnan(volume_sma_values[-1]):
            continue

        # Get the latest indicator values for the current candle
        latest_ema = ema_values[-1]
        latest_atr = atr_values[-1]
        latest_adx = adx_values[-1]
        latest_plus_di = plus_di_values[-1]
        latest_minus_di = minus_di_values[-1]
        latest_close = close_prices[-1]
        latest_volume = volumes[-1]
        latest_volume_sma = volume_sma_values[-1]

        latest_timestamp = warm_candles[-1].hour
        latest_price_for_signal = warm_candles[-1].close

        # Calculate Keltner Bands for the latest candle
        upper_keltner_band = latest_ema + (ATR_Multiplier * latest_atr)
        lower_keltner_band = latest_ema - (ATR_Multiplier * latest_atr)

        # Buy Signal condition
        if latest_close > upper_keltner_band \
           and latest_adx > ADX_Threshold \
           and latest_plus_di > latest_minus_di \
           and latest_volume > (latest_volume_sma * Volume_Multiplier):
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_timestamp,
                price=latest_price_for_signal,
                rule_id="266701ac-c78c-48ec-8000-2075c06d90e5"
            ))

        # Sell Signal condition
        elif latest_close < lower_keltner_band \
             and latest_adx > ADX_Threshold \
             and latest_minus_di > latest_plus_di \
             and latest_volume > (latest_volume_sma * Volume_Multiplier):
            signals.append(SellSignal(
                pair=pair,
                timestamp=latest_timestamp,
                price=latest_price_for_signal,
                rule_id="266701ac-c78c-48ec-8000-2075c06d90e5"
            ))

    return signals