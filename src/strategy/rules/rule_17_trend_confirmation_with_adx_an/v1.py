from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Parameters ---
SHORT_MA_PERIOD = 10
LONG_MA_PERIOD = 30
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ATR_PERIOD = 14
VOLATILITY_MULTIPLIER = 1.5

# Rule ID for signals
RULE_ID = "76c395b8-fa99-47a0-aac9-69c467088e68"

# Minimum candles required for all indicators to be calculated
# Longest period is LONG_MA_PERIOD = 30.
# ADX(N) requires approximately 3*N - 1 candles. For N=14, this is 3*14 - 1 = 41 candles.
# Recent_avg_ATR (SMA(ATR, 2*N)) requires approximately 3*N candles. For N=14, this is 3*14 = 42 candles.
# So, the maximum requirement is 42 candles.
MIN_CANDLES_REQUIRED = 42

# --- Technical Indicator Implementations ---

def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < period:
        return np.array([])
    # Using convolution for efficiency
    weights = np.ones(period) / period
    return np.convolve(data, weights, 'valid')

def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Calculates True Range (TR).
    TR = max(high - low, abs(high - previous_close), abs(low - previous_close))
    """
    if len(high) < 2:
        return np.array([])
    
    tr = np.zeros(len(high) - 1)
    for i in range(1, len(high)):
        tr[i-1] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return tr

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculates Average True Range (ATR) using SMA of TR."""
    tr_values = _true_range(high, low, close)
    if len(tr_values) < period:
        return np.array([])
    
    return _sma(tr_values, period)

def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculates Average Directional Index (ADX)."""
    if len(high) <= period * 2: # Minimum data for ADX calculation (rough estimate, more precise below)
        return np.array([])

    # DM+ and DM-
    # These are calculated based on current vs previous High/Low
    # So they align with the candle from index 1 onwards.
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # True Range (TR) from the second candle onwards, same length as DM+ / DM-
    tr_values = _true_range(high, low, close)

    # Check if we have enough TR values for the first SMA
    if len(tr_values) < period:
        return np.array([])

    # Calculate SMAs of DM+, DM- and TR
    smooth_plus_dm = _sma(plus_dm, period)
    smooth_minus_dm = _sma(minus_dm, period)
    smooth_tr = _sma(tr_values, period)

    # Ensure all smoothed arrays are of the same length and non-empty
    if len(smooth_plus_dm) == 0 or len(smooth_minus_dm) == 0 or len(smooth_tr) == 0:
        return np.array([])
    
    # Handle cases where smooth_tr is zero to prevent division by zero
    # If smooth_tr is zero, then there's no price movement, so DI+ / DI- are undefined or zero.
    # In such cases, ADX is effectively zero or undefined.
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = (smooth_plus_dm / smooth_tr) * 100
        di_minus = (smooth_minus_dm / smooth_tr) * 100
    
    di_plus[np.isnan(di_plus)] = 0
    di_minus[np.isnan(di_minus)] = 0
    
    # Calculate DX
    di_sum = di_plus + di_minus
    # Handle cases where di_sum is zero to prevent division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = np.abs(di_plus - di_minus) / di_sum * 100
    
    dx[np.isnan(dx)] = 0 # If di_sum is 0, dx is 0
    dx[np.isinf(dx)] = 0 # Should not happen with the isnan check

    # Calculate ADX (SMA of DX)
    adx_values = _sma(dx, period)

    return adx_values

def _crossover(series_a: np.ndarray, series_b: np.ndarray) -> tuple[bool, bool]:
    """
    Checks for a crossover in the last period.
    Returns (bullish_crossover, bearish_crossover).
    A bullish crossover is when series_a crosses above series_b.
    A bearish crossover is when series_b crosses above series_a.
    """
    if len(series_a) < 2 or len(series_b) < 2:
        return False, False
    
    # Check if series_a crosses above series_b
    bullish = (series_a[-2] <= series_b[-2]) and (series_a[-1] > series_b[-1])
    # Check if series_b crosses above series_a
    bearish = (series_b[-2] <= series_a[-2]) and (series_b[-1] > series_a[-1])
    
    return bullish, bearish

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm

        if len(candles) < MIN_CANDLES_REQUIRED:
            # Not enough historical data for full indicator calculation
            continue

        # Extract required data, ensuring it's sorted by time (oldest first)
        # WarmCandle list is assumed to be sorted by 'hour' ascending.
        close_prices = np.array([c.close for c in candles], dtype=float)
        high_prices = np.array([c.high for c in candles], dtype=float)
        low_prices = np.array([c.low for c in candles], dtype=float)

        # Calculate indicators
        short_ma = _sma(close_prices, SHORT_MA_PERIOD)
        long_ma = _sma(close_prices, LONG_MA_PERIOD)
        adx_values = _adx(high_prices, low_prices, close_prices, ADX_PERIOD)
        atr_values = _atr(high_prices, low_prices, close_prices, ATR_PERIOD)

        # Check for sufficient data for the *latest* values of all indicators
        # If any indicator array is empty, or too short to get the last value, skip.
        if (len(short_ma) < 2 or len(long_ma) < 2 or # Need 2 for crossover check
                len(adx_values) == 0 or len(atr_values) == 0):
            continue

        # Get the latest values for comparison
        current_adx = adx_values[-1]
        current_atr = atr_values[-1]

        # Calculate recent_avg_atr
        # Requires at least ATR_PERIOD * 2 ATR values to calculate its SMA
        if len(atr_values) < ATR_PERIOD * 2:
            continue
        
        recent_avg_atr = _sma(atr_values, ATR_PERIOD * 2)
        if len(recent_avg_atr) == 0:
            continue
        current_recent_avg_atr = recent_avg_atr[-1]

        # Check for MA crossover (using the last two aligned MA values)
        bullish_crossover, bearish_crossover = _crossover(short_ma, long_ma)

        # Check for trend strength and volatility filter
        is_trending_strong = (current_adx > ADX_THRESHOLD)
        is_volatility_normal = (current_atr < (current_recent_avg_atr * VOLATILITY_MULTIPLIER))

        # Generate signals
        latest_candle = candles[-1]
        if bullish_crossover and is_trending_strong and is_volatility_normal:
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_candle.hour,
                price=latest_candle.close,
                rule_id=RULE_ID
            ))
        elif bearish_crossover and is_trending_strong and is_volatility_normal:
            signals.append(SellSignal(
                pair=pair,
                timestamp=latest_candle.hour,
                price=latest_candle.close,
                rule_id=RULE_ID
            ))
            
    return signals