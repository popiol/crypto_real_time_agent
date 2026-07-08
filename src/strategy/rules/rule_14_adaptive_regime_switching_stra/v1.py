from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule Configuration ---
RULE_ID = "33a3d2a5-fefb-4559-b260-0f3fb925379e"

# ADX parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25

# Bollinger Bands parameters
BB_PERIOD = 20
BB_STD_DEV = 2.0

# Moving Average Crossover parameters
SHORT_MA_PERIOD = 10
LONG_MA_PERIOD = 30

# Minimum candles required for calculations
# Max of:
# - LONG_MA_PERIOD + 1 (for MA crossover to have at least two points for comparison)
# - BB_PERIOD (for Bollinger Bands, only need last value, so BB_PERIOD candles for BB_PERIOD SMA)
# - 2 * ADX_PERIOD (for ADX, to get at least one stable ADX value after double smoothing)
MIN_CANDLES = max(LONG_MA_PERIOD + 1, BB_PERIOD, 2 * ADX_PERIOD)


# --- Helper functions for indicator calculations using numpy ---

def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < period:
        return np.array([])
    # np.convolve with 'valid' mode returns only the parts where the kernel fully overlaps.
    # This aligns the output to the end of the window.
    return np.convolve(data, np.ones(period), 'valid') / period

def _std_dev(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Rolling Standard Deviation."""
    if len(data) < period:
        return np.array([])
    res = np.zeros(len(data) - period + 1)
    for i in range(len(res)):
        res[i] = np.std(data[i : i + period])
    return res

def _wilder_smoothing(data: np.ndarray, period: int) -> np.ndarray:
    """Applies Wilder's smoothing (similar to EMA) to data.
    The first value is an SMA of the initial 'period' elements,
    subsequent values use an EMA-like formula:
    smoothed_val_i = (prev_smoothed_val * (period - 1) + current_raw_val) / period
    """
    if len(data) < period:
        return np.array([])
    
    smoothed = np.zeros(len(data) - period + 1)
    # First value is SMA of the first 'period' elements
    smoothed[0] = np.mean(data[:period])

    for i in range(1, len(smoothed)):
        # The current_raw_val corresponds to data[i + period - 1] because 'smoothed' array is shorter than 'data'
        smoothed[i] = (smoothed[i-1] * (period - 1) + data[i + period - 1]) / period
    return smoothed

def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculates Average Directional Index (ADX) using Wilder's smoothing."""
    # Need at least `period * 2` candles for reliable ADX after double smoothing
    # (period for TR/DM smoothing, then another period for DX smoothing)
    if len(high) < period * 2:
        return np.array([])

    # 1. Calculate True Range (TR), Positive Directional Movement (+DM), and Negative Directional Movement (-DM)
    # for each bar (from 1 to N-1). These arrays will have length len(high) - 1.
    tr_raw = np.zeros(len(high) - 1)
    plus_dm_raw = np.zeros(len(high) - 1)
    minus_dm_raw = np.zeros(len(high) - 1)

    for i in range(1, len(high)):
        tr_raw[i-1] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm_raw[i-1] = up_move
        else:
            plus_dm_raw[i-1] = 0

        if down_move > up_move and down_move > 0:
            minus_dm_raw[i-1] = down_move
        else:
            minus_dm_raw[i-1] = 0
    
    # 2. Smooth TR, +DM, -DM using Wilder's smoothing
    # These smoothed arrays will have length (len(high) - 1) - period + 1 = len(high) - period
    smoothed_tr = _wilder_smoothing(tr_raw, period)
    smoothed_plus_dm = _wilder_smoothing(plus_dm_raw, period)
    smoothed_minus_dm = _wilder_smoothing(minus_dm_raw, period)
    
    if len(smoothed_tr) == 0: # This check should ideally be covered by the initial len(high) check
        return np.array([])

    # 3. Calculate +DI and -DI
    # Avoid division by zero for DI calculations
    plus_di = np.where(smoothed_tr != 0, 100 * smoothed_plus_dm / smoothed_tr, 0)
    minus_di = np.where(smoothed_tr != 0, 100 * smoothed_minus_dm / smoothed_tr, 0)

    # 4. Calculate DX (Directional Movement Index)
    di_sum = plus_di + minus_di
    # Avoid division by zero for DX calculation
    dx_values = np.where(di_sum != 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0)

    # 5. Smooth DX to get ADX
    # This ADX array will have length (len(high) - period) - period + 1 = len(high) - 2*period + 1
    adx_result = _wilder_smoothing(dx_values, period)

    return adx_result


# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Adaptive Regime-Switching Strategy (Bollinger/MA Crossover).

    This rule dynamically switches between a mean-reversion strategy (Bollinger Bands)
    and a trend-following strategy (Moving Average Crossover) based on the market's
    prevailing regime, as determined by the Average Directional Index (ADX).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm
        ticks = pair_data.hot

        # Ensure we have enough historical data (candles) for all calculations
        # and at least one tick for current price/timestamp.
        if len(candles) < MIN_CANDLES or not ticks:
            continue

        # Extract price arrays from candles
        high_prices = np.array([c.high for c in candles], dtype=float)
        low_prices = np.array([c.low for c in candles], dtype=float)
        close_prices = np.array([c.close for c in candles], dtype=float)
        
        current_price = ticks[-1].last_price
        signal_timestamp = ticks[-1].polled_at

        # --- Calculate Indicators ---

        # ADX
        adx_values = _adx(high_prices, low_prices, close_prices, ADX_PERIOD)
        if len(adx_values) == 0:
            continue # Not enough data to calculate ADX
        current_adx = adx_values[-1]

        # Bollinger Bands
        sma_bb = _sma(close_prices, BB_PERIOD)
        std_bb = _std_dev(close_prices, BB_PERIOD)
        if len(sma_bb) == 0 or len(std_bb) == 0:
            continue # Not enough data for Bollinger Bands

        # The last elements of sma_bb and std_bb correspond to the latest candle
        current_sma_bb = sma_bb[-1]
        current_std_bb = std_bb[-1]

        upper_band = current_sma_bb + (current_std_bb * BB_STD_DEV)
        lower_band = current_sma_bb - (current_std_bb * BB_STD_DEV)

        # Moving Averages for Crossover
        sma_short = _sma(close_prices, SHORT_MA_PERIOD)
        sma_long = _sma(close_prices, LONG_MA_PERIOD)
        
        # Need at least two values for both MAs to detect a crossover
        if len(sma_short) < 2 or len(sma_long) < 2:
            continue # Not enough data for MA crossover

        # The last two elements of each MA array represent the current and previous candle's MA value
        current_sma_short = sma_short[-1]
        prev_sma_short = sma_short[-2]
        current_sma_long = sma_long[-1]
        prev_sma_long = sma_long[-2]

        # --- Regime Switching Logic ---
        if current_adx < ADX_THRESHOLD:
            # Ranging Market (Mean Reversion Strategy using Bollinger Bands)
            if current_price < lower_band:
                signals.append(BuySignal(
                    pair=pair,
                    timestamp=signal_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=0.7 # Example confidence for ranging market buy
                ))
            elif current_price > upper_band:
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=signal_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=0.7 # Example confidence for ranging market sell
                ))
        elif current_adx >= ADX_THRESHOLD: # Using >= to include the threshold value in trending regime
            # Trending Market (Trend Following Strategy using MA Crossover)
            # Buy signal: Short MA crosses above Long MA
            if current_sma_short > current_sma_long and prev_sma_short <= prev_sma_long:
                signals.append(BuySignal(
                    pair=pair,
                    timestamp=signal_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=0.8 # Example confidence for trending market buy
                ))
            # Sell signal: Short MA crosses below Long MA
            elif current_sma_short < current_sma_long and prev_sma_short >= prev_sma_long:
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=signal_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=0.8 # Example confidence for trending market sell
                ))

    return signals