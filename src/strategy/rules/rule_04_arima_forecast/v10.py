from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle


# Minimum warm candles needed for a stable ARIMA fit.
# While ARIMA(1,1,0) can technically run with 3 prices, 10 is more robust.
MIN_ARIMA_PRICES = 10 

# Forecast horizon in hours
FORECAST_HORIZON = 3

# ATR parameters
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5

# Multi-Timeframe RSI Confirmation Parameters
HIGHER_TIMEFRAME_FACTOR = 4  # e.g., 4x current timeframe (4-hour candles if current is 1-hour)
# RSI period for the higher timeframe.
# Note: The `warm` data (max 24 hourly candles) limits the number of aggregated
# 4-hour candles to 6 (24 / 4). A standard RSI(14) would require at least 15 candles.
# We adjust this value to `5` to allow calculation with the available data.
RSI_PERIOD_HIGHER_TF = 5
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70

# Overall minimum warm candles required for all calculations to proceed:
# 1. ARIMA: MIN_ARIMA_PRICES (e.g., 10)
# 2. ATR: ATR_PERIOD + 1 (e.g., 14 + 1 = 15)
# 3. Higher TF RSI: (RSI_PERIOD_HIGHER_TF + 1) * HIGHER_TIMEFRAME_FACTOR (e.g., (5 + 1) * 4 = 24)
MIN_WARM_CANDLES_FOR_SIGNAL = max(
    MIN_ARIMA_PRICES, 
    ATR_PERIOD + 1, 
    (RSI_PERIOD_HIGHER_TF + 1) * HIGHER_TIMEFRAME_FACTOR
)


def _fit_arima110(prices: list[float]) -> tuple[float, float, float]:
    """Fit ARIMA(1,1,0) to a price series via OLS on first differences.

    Returns (phi, intercept, sigma_residual) where phi is the AR(1) coefficient
    on the differenced series and sigma_residual is in price units.
    """
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    x = diffs[:-1]
    y = diffs[1:]
    n = len(x)

    if n < 2: # Need at least 2 differences for x and y to fit, meaning 3 prices
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
    # sigma calculation needs at least 2 residuals for stdev. If n=2, len(residuals)=2.
    sigma = statistics.stdev(residuals) if n >= 3 else abs(residuals[0]) if residuals else 0.0

    return phi, intercept, sigma


def _forecast(prices: list[float], phi: float, intercept: float, horizon: int) -> float:
    """Iterate the ARIMA(1,1,0) recursion h steps ahead."""
    if len(prices) < 2: # Need at least 2 prices to calculate a diff
        return prices[-1] if prices else 0.0

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


def _calculate_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int
) -> float | None:
    """Calculates the Average True Range (ATR)."""
    if not (len(highs) == len(lows) == len(closes)) or len(highs) < period + 1:
        # Need at least 'period + 1' candles to calculate the first 'period' True Ranges
        return None

    true_ranges: list[float] = []
    # TR calculation starts from the second candle (index 1) as it needs prev_close
    for i in range(1, len(highs)):
        prev_close = closes[i - 1]
        current_high = highs[i]
        current_low = lows[i]

        tr1 = current_high - current_low
        tr2 = abs(current_high - prev_close)
        tr3 = abs(current_low - prev_close)

        true_ranges.append(max(tr1, tr2, tr3))

    # At this point, len(true_ranges) should be len(highs) - 1.
    if len(true_ranges) < period: # This should be covered by the initial check, but defensive
        return None 

    # Calculate initial ATR as the simple average of the first 'period' True Ranges
    initial_atr = sum(true_ranges[:period]) / period

    # Apply Wilder's smoothing for subsequent periods to get the latest ATR
    atr_value = initial_atr
    for i in range(period, len(true_ranges)):
        atr_value = ((atr_value * (period - 1)) + true_ranges[i]) / period

    return atr_value


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # 1. Extract data
        # Current timeframe data (hourly candles)
        current_tf_closes = [c.close for c in pair_data.warm]
        current_tf_highs = [c.high for c in pair_data.warm]
        current_tf_lows = [c.low for c in pair_data.warm]
        
        # Higher timeframe data (e.g., 4-hour candles from hourly data)
        higher_tf_closes = _aggregate_closes_by_factor(pair_data.warm, HIGHER_TIMEFRAME_FACTOR)

        # 2. Check for sufficient data
        if len(pair_data.warm) < MIN_WARM_CANDLES_FOR_SIGNAL:
            continue
        if not pair_data.hot: # Need hot data for current price and timestamp
            continue

        # 3. Calculate ARIMA(1,1,0) forecast for the next period
        phi, intercept, sigma = _fit_arima110(current_tf_closes)

        # Skip if ARIMA model is degenerate or unstable (e.g., all prices are identical)
        if sigma == 0 or not math.isfinite(phi) or not math.isfinite(intercept):
            continue

        forecast_price = _forecast(current_tf_closes, phi, intercept, FORECAST_HORIZON)
        
        # 4. Determine current_price and timestamp
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # 5. Calculate ATR-adjusted deviation_threshold
        atr = _calculate_atr(current_tf_highs, current_tf_lows, current_tf_closes, ATR_PERIOD)
        
        # Avoid signals based on zero or near-zero volatility
        if atr is None or atr <= 0:
            continue
        
        deviation_threshold = ATR_MULTIPLIER * atr
        
        # 6. Calculate higher_timeframe_rsi
        higher_timeframe_rsi = _calculate_rsi(higher_tf_closes, RSI_PERIOD_HIGHER_TF)
        if higher_timeframe_rsi is None:
            # This case should be covered by MIN_WARM_CANDLES_FOR_SIGNAL,
            # but it's a defensive check if data somehow gets corrupted.
            continue 

        # 7-10. Define signal conditions based on ARIMA forecast and RSI confirmation
        buy_signal_condition_arima = forecast_price > (current_price + deviation_threshold)
        sell_signal_condition_arima = forecast_price < (current_price - deviation_threshold)

        buy_signal_condition_rsi = higher_timeframe_rsi < RSI_OVERSOLD_THRESHOLD
        sell_signal_condition_rsi = higher_timeframe_rsi > RSI_OVERBOUGHT_THRESHOLD

        # 11-12. Combine conditions for final signal
        if buy_signal_condition_arima and buy_signal_condition_rsi:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif sell_signal_condition_arima and sell_signal_condition_rsi:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals