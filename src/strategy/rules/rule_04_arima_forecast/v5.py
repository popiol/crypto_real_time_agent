from __future__ import annotations

import math
import statistics
import numpy as np
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle


# Minimum warm candles needed for a reliable fit for ARIMA
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Lookback window for calculating recent volatility (in hours/candles)
LOOKBACK_VOLATILITY_WINDOW = 24

# Multiplier for the historical standard deviation to determine the deviation threshold
VOLATILITY_MULTIPLIER = 1.5

# --- Candlestick Pattern Parameters ---
# A candle body is considered "small" if its size is less than this fraction of the total candle range (high - low).
BODY_PERCENT_OF_RANGE_SMALL = 0.20
# For Hammer/Hanging Man, the lower shadow must be at least this many times the body size.
SHADOW_BODY_RATIO_LONG = 2.0
# For Hammer/Hanging Man, the upper shadow must be less than this fraction of the body size.
SHADOW_PERCENT_OF_BODY_SMALL = 0.25 # e.g., upper shadow < 25% of body


def _get_candle_body_size(open_price: float, close_price: float) -> float:
    """Calculates the absolute size of the candle body."""
    return abs(close_price - open_price)

def _is_hammer(c: WarmCandle) -> bool:
    """Detects a Hammer candlestick pattern.
    A Hammer has a small body, a long lower shadow, and little or no upper shadow.
    It can be bullish or bearish in color, but the body is at the upper end of the range.
    """
    body = _get_candle_body_size(c.open_price, c.close)
    candle_range = c.high - c.low

    if candle_range == 0: # Avoid division by zero for range-based checks
        return False
    
    # 1. Small body relative to the candle's total range
    if body / candle_range >= BODY_PERCENT_OF_RANGE_SMALL:
        return False

    # Determine actual open/close for shadow calculation
    real_open = c.open_price
    real_close = c.close

    lower_shadow = min(real_open, real_close) - c.low
    upper_shadow = c.high - max(real_open, real_close)

    # 2. Long lower shadow (at least SHADOW_BODY_RATIO_LONG times the body)
    #    Require a non-zero body for ratio calculation.
    if body > 0.0001: # Using a small epsilon to check for non-zero body
        if lower_shadow / body < SHADOW_BODY_RATIO_LONG:
            return False
    else: # If body is extremely small (doji-like), check lower shadow against total range
        if lower_shadow < (candle_range * 0.5): # e.g., lower shadow should be at least half the range
            return False

    # 3. Little or no upper shadow (upper shadow less than SHADOW_PERCENT_OF_BODY_SMALL of body)
    if body > 0.0001:
        if upper_shadow / body >= SHADOW_PERCENT_OF_BODY_SMALL:
            return False
    else: # If body is extremely small (doji-like), check upper shadow against total range
        if upper_shadow > (candle_range * BODY_PERCENT_OF_RANGE_SMALL * 0.5): # upper shadow small relative to range
            return False

    return True


def _is_bullish_engulfing(c0: WarmCandle, c1: WarmCandle) -> bool:
    """Detects a Bullish Engulfing candlestick pattern."""
    # c0 (previous candle) must be bearish
    if c0.close >= c0.open_price:
        return False

    # c1 (current candle) must be bullish
    if c1.close <= c1.open_price:
        return False

    # c1's body must completely engulf c0's body
    if c1.open_price < c0.close and c1.close > c0.open_price:
        return True
    return False


def _is_hanging_man(c: WarmCandle) -> bool:
    """Detects a Hanging Man candlestick pattern.
    The pattern is structurally identical to a Hammer, but its significance
    is as a bearish reversal when it appears after an uptrend.
    For this rule, we just check the pattern itself.
    """
    # It's the same structural pattern as a Hammer
    return _is_hammer(c)


def _is_bearish_engulfing(c0: WarmCandle, c1: WarmCandle) -> bool:
    """Detects a Bearish Engulfing candlestick pattern."""
    # c0 (previous candle) must be bullish
    if c0.close <= c0.open_price:
        return False

    # c1 (current candle) must be bearish
    if c1.close >= c1.open_price:
        return False

    # c1's body must completely engulf c0's body
    if c1.open_price > c0.close and c1.close < c0.open_price:
        return True
    return False

def _is_bullish_reversal(candles: list[WarmCandle]) -> bool:
    """Checks for any bullish reversal pattern among the last candles."""
    if not candles:
        return False
    
    # Check for Hammer (single candle)
    if _is_hammer(candles[-1]):
        return True
    
    # Check for Bullish Engulfing (requires at least two candles)
    if len(candles) >= 2 and _is_bullish_engulfing(candles[-2], candles[-1]):
        return True
        
    return False

def _is_bearish_reversal(candles: list[WarmCandle]) -> bool:
    """Checks for any bearish reversal pattern among the last candles."""
    if not candles:
        return False
        
    # Check for Hanging Man (single candle)
    if _is_hanging_man(candles[-1]): # Structurally same as hammer
        return True

    # Check for Bearish Engulfing (requires at least two candles)
    if len(candles) >= 2 and _is_bearish_engulfing(candles[-2], candles[-1]):
        return True
        
    return False

# --- End Candlestick Pattern Functions ---

def _fit_arima110(prices: list[float]) -> tuple[float, float, float]:
    """Fit ARIMA(1,1,0) to a price series via OLS on first differences.

    Returns (phi, intercept, sigma_residual) where phi is the AR(1) coefficient
    on the differenced series and sigma_residual is in price units.
    """
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    x = diffs[:-1]
    y = diffs[1:]
    n = len(x)

    if n < 2:
        return 0.0, 0.0, 0.0

    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)

    cov_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    var_x = sum((xi - x_mean) ** 2 for xi in x)

    if var_x == 0:
        phi = 0.0
        intercept = y_mean
    else:
        phi = cov_xy / var_x
        intercept = y_mean - phi * x_mean

    residuals = [yi - (phi * xi + intercept) for xi, yi in zip(x, y)]
    sigma = statistics.stdev(residuals) if n >= 3 else abs(residuals[0]) if residuals else 0.0

    return phi, intercept, sigma


def _forecast(prices: list[float], phi: float, intercept: float, horizon: int) -> float:
    """Iterate the ARIMA(1,1,0) recursion h steps ahead."""
    if len(prices) < 2: # Need at least two prices to calculate last_diff
        return prices[-1] if prices else 0.0 # Or raise an error, depending on desired behavior
        
    last_diff = prices[-1] - prices[-2]
    price = prices[-1]
    delta = last_diff
    for _ in range(horizon):
        delta = intercept + phi * delta
        price += delta
    return price


def _forecast_std(phi: float, sigma: float, horizon: int) -> float:
    """Exact h-step forecast std for ARIMA(1,1,0).

    ψ_k = 1 + φ + φ² + … + φ^k  (impulse-response weights)
    Var(e_h) = σ² · Σ_{k=0}^{h-1} ψ_k²
    """
    variance = 0.0
    for k in range(horizon):
        psi_k = sum(phi**j for j in range(k + 1))
        variance += psi_k**2
    return sigma * math.sqrt(variance)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm candles for ARIMA and candlestick patterns
        # Candlestick patterns might need 2 candles for multi-candle patterns.
        # ARIMA needs MIN_CANDLES. So, we need at least max(MIN_CANDLES, 2) candles.
        if len(pair_data.warm) < max(MIN_CANDLES, 2) or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        
        # Ensure enough data for volatility calculation
        if len(prices) < LOOKBACK_VOLATILITY_WINDOW:
            continue

        # 1. Calculate ARIMA(1,1,0) forecast
        phi, intercept, sigma = _fit_arima110(prices)

        if sigma == 0 or not math.isfinite(phi):
            continue

        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # 2. Calculate adaptive deviation threshold based on recent volatility
        recent_prices_for_volatility = prices[-LOOKBACK_VOLATILITY_WINDOW:]
        
        price_std = np.std(recent_prices_for_volatility)
        
        deviation_threshold = VOLATILITY_MULTIPLIER * price_std
        
        # If volatility is zero or near-zero, the threshold would be zero, making any
        # tiny deviation a signal. We might want to avoid this or set a minimum.
        if deviation_threshold <= 0:
            continue

        # 3. Determine forecast signal and price difference
        price_diff = forecast_price - current_price
        
        # 4. Add candlestick reversal confirmation
        # Use the warm candles for pattern detection
        current_candlesticks = pair_data.warm 

        if price_diff > deviation_threshold: # ARIMA suggests a potential buy
            if _is_bullish_reversal(current_candlesticks):
                signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif price_diff < -deviation_threshold: # ARIMA suggests a potential sell
            if _is_bearish_reversal(current_candlesticks):
                signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals