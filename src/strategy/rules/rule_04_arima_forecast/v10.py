from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from collections import deque

import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# Minimum warm candles needed for a reliable fit for current timeframe ARIMA
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Multi-Timeframe RSI Confirmation Parameters
HIGHER_TIMEFRAME_FACTOR = 4  # e.g., 4x current timeframe (4-hour candles if current is 1-hour)

# RSI period for the higher timeframe.
# Note: The `warm` data (max 24 hourly candles) limits the number of aggregated
# 4-hour candles to 6 (24 / 4). A standard RSI(14) would require at least 15 candles.
# We adjust this value to `5` to allow calculation with the available data.
RSI_PERIOD_HIGHER_TF = 5
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70

# ATR Parameters
ATR_PERIOD = 14 # Standard ATR period. Max 24 warm candles allows for 24 - 14 = 10 ATR values.
VOLATILITY_REGIME_PERIOD = 10 # Period to assess recent ATRs for volatility regime. Max 10 based on ATR_PERIOD and warm candles.
VOLATILITY_STD_THRESHOLD = 0.5 # Multiplier for standard deviation of ATRs to define high/low regimes.
HIGH_VOLATILITY_ATR_MULTIPLIER = 2.0 # ATR multiplier for high volatility regime.
LOW_VOLATILITY_ATR_MULTIPLIER = 1.0 # ATR multiplier for low volatility regime.
NORMAL_VOLATILITY_ATR_MULTIPLIER = 1.5 # ATR multiplier for normal volatility regime.


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
    if len(prices) < 2:
        return prices[-1] if prices else 0.0

    last_diff = prices[-1] - prices[-2]
    price = prices[-1]
    delta = last_diff
    for _ in range(horizon):
        delta = intercept + phi * delta
        price += delta
    return price


def _aggregate_closes_by_factor(candles: list[WarmCandle], factor: int) -> list[float]:
    """Aggregates close prices of WarmCandle (hourly) into higher timeframe blocks.
    Takes the close price of the last candle in each complete block of `factor` candles.
    """
    if not candles or factor <= 0:
        return []

    aggregated_closes = []
    # Iterate over full blocks of `factor` candles
    for i in range(len(candles) // factor):
        # The close of the aggregated candle is the close of the last candle in the block
        aggregated_closes.append(candles[(i + 1) * factor - 1].close)
    
    return aggregated_closes


def _calculate_rsi(prices: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI) for a given price series."""
    if len(prices) < period + 1:
        return None

    # Calculate initial gains and losses
    gains = [0.0] * (len(prices) - 1)
    losses = [0.0] * (len(prices) - 1)

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains[i - 1] = change
        else:
            losses[i - 1] = abs(change)

    # Calculate initial average gain and loss over the first 'period' changes
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Calculate initial RS and RSI
    if avg_loss == 0:
        rs = float('inf')  # All gains, no losses -> RSI 100
    else:
        rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # Apply Wilder's smoothing for subsequent periods to get the latest RSI
    for i in range(period, len(gains)): # Iterate through the remaining changes
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

        if avg_loss == 0:
            rs = float('inf')
        else:
            rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    return rsi

def _calculate_true_range(high: float, low: float, prev_close: float) -> float:
    """Calculates the True Range for a given candle."""
    return max(high - low, abs(high - prev_close), abs(low - prev_close))

def _calculate_all_atrs(candles: list[WarmCandle], period: int) -> list[float] | None:
    """Calculates a list of ATR values for a series of candles.
    Returns a list of ATRs, where the i-th element is the ATR for the candle at index (i + period).
    """
    if len(candles) < period + 1:
        return None

    true_ranges = []
    # Calculate initial True Ranges
    for i in range(1, len(candles)):
        tr = _calculate_true_range(candles[i].high, candles[i].low, candles[i-1].close)
        true_ranges.append(tr)

    # Calculate initial ATR
    # The first 'period' true ranges are used for the initial ATR
    # This ATR corresponds to the candle at index `period`
    current_atr = sum(true_ranges[:period]) / period
    atrs = [current_atr]

    # Apply Wilder's smoothing for subsequent ATRs
    for i in range(period, len(true_ranges)):
        current_atr = ((current_atr * (period - 1)) + true_ranges[i]) / period
        atrs.append(current_atr)
    
    return atrs


def _calculate_dynamic_atr_threshold(candles: list[WarmCandle]) -> float | None:
    """Calculates a dynamically scaled ATR threshold based on market volatility regime."""
    
    all_atrs = _calculate_all_atrs(candles, ATR_PERIOD)

    if all_atrs is None or len(all_atrs) < VOLATILITY_REGIME_PERIOD:
        return None

    current_atr = all_atrs[-1]
    
    # Get recent ATRs for volatility regime analysis
    recent_atrs = all_atrs[-VOLATILITY_REGIME_PERIOD:]

    if len(recent_atrs) < 1: # Should be caught by previous check, but defensive
        return None
    
    mean_atr = statistics.mean(recent_atrs)
    
    # Handle cases where std_dev cannot be calculated (e.g., all recent_atrs are identical)
    std_dev_atr = 0.0
    if len(recent_atrs) >= 2:
        try:
            std_dev_atr = statistics.stdev(recent_atrs)
        except statistics.StatisticsError:
            std_dev_atr = 0.0 # All values are the same, stdev is 0

    threshold_multiplier: float

    if std_dev_atr == 0 or current_atr == 0: # Avoid division by zero or static market
        threshold_multiplier = NORMAL_VOLATILITY_ATR_MULTIPLIER
    elif current_atr > mean_atr + VOLATILITY_STD_THRESHOLD * std_dev_atr:
        # High volatility regime
        threshold_multiplier = HIGH_VOLATILITY_ATR_MULTIPLIER
    elif current_atr < mean_atr - VOLATILITY_STD_THRESHOLD * std_dev_atr:
        # Low volatility regime
        threshold_multiplier = LOW_VOLATILITY_ATR_MULTIPLIER
    else:
        # Normal volatility regime (interpolate or use default)
        threshold_multiplier = NORMAL_VOLATILITY_ATR_MULTIPLIER
    
    return threshold_multiplier * current_atr


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Current timeframe data (hourly candles)
        current_tf_prices = [c.close for c in pair_data.warm]
        
        # Higher timeframe data (e.g., 4-hour candles from hourly data)
        higher_tf_prices = _aggregate_closes_by_factor(pair_data.warm, HIGHER_TIMEFRAME_FACTOR)

        # 2. If insufficient data, return "NO_SIGNAL"
        # Check for sufficient data for ARIMA forecast
        if len(current_tf_prices) < MIN_CANDLES:
            continue
        
        # Check for sufficient data for dynamic ATR threshold calculation
        # _calculate_all_atrs needs ATR_PERIOD + 1 candles for the first ATR.
        # It returns `len(candles) - ATR_PERIOD` ATR values.
        # We need at least VOLATILITY_REGIME_PERIOD number of ATRs to calculate mean/stdev.
        # So, `len(candles) - ATR_PERIOD >= VOLATILITY_REGIME_PERIOD`
        # which implies `len(candles) >= ATR_PERIOD + VOLATILITY_REGIME_PERIOD`.
        if len(pair_data.warm) < ATR_PERIOD + VOLATILITY_REGIME_PERIOD:
            continue

        # Check for sufficient data for higher timeframe RSI calculation
        if len(higher_tf_prices) < RSI_PERIOD_HIGHER_TF + 1:
            continue

        # Need hot data for current price and timestamp
        if not pair_data.hot:
            continue

        # 3. Calculate ARIMA(arima_order) forecast for the next period
        phi, intercept, sigma = _fit_arima110(current_tf_prices)

        # Skip if ARIMA model is degenerate
        if sigma == 0 or not math.isfinite(phi) or not math.isfinite(intercept):
            continue

        forecast_price = _forecast(current_tf_prices, phi, intercept, FORECAST_HORIZON)
        
        # 4. Determine current_price
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # 5. Calculate dynamic_threshold
        dynamic_threshold = _calculate_dynamic_atr_threshold(pair_data.warm)
        
        # Avoid signals if threshold calculation failed or resulted in zero
        if dynamic_threshold is None or dynamic_threshold <= 0:
            continue

        # 6. Calculate higher_timeframe_rsi
        higher_timeframe_rsi = _calculate_rsi(higher_tf_prices, RSI_PERIOD_HIGHER_TF)
        if higher_timeframe_rsi is None:
            continue # Not enough data for RSI calculation (should be caught by earlier check, but good to double-check)

        # 7-10. Define signal conditions based on ARIMA forecast and RSI confirmation
        buy_signal_condition_arima = forecast_price > (current_price + dynamic_threshold)
        sell_signal_condition_arima = forecast_price < (current_price - dynamic_threshold)

        buy_signal_condition_rsi = higher_timeframe_rsi < RSI_OVERSOLD_THRESHOLD
        sell_signal_condition_rsi = higher_timeframe_rsi > RSI_OVERBOUGHT_THRESHOLD

        # 11-12. Combine conditions for final signal
        if buy_signal_condition_arima and buy_signal_condition_rsi:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif sell_signal_condition_arima and sell_signal_condition_rsi:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals