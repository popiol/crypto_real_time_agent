from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta

import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle


# Minimum warm candles needed for a reliable fit for current timeframe ARIMA
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Lookback window for calculating recent volatility (in hours/candles)
LOOKBACK_VOLATILITY_WINDOW = 24

# Multiplier for the historical standard deviation to determine the deviation threshold
VOLATILITY_MULTIPLIER = 1.5

# Multi-Timeframe RSI Confirmation Parameters
HIGHER_TIMEFRAME_FACTOR = 4  # e.g., 4x current timeframe (4-hour candles if current is 1-hour)

# RSI period for the higher timeframe.
# NOTE: Adjusted from 5 to 2. The original RSI_PERIOD_HIGHER_TF=5, with max 6 higher-timeframe
# candles (from 24 hourly), would only yield 1 valid RSI value. This makes percentile
# calculations for adaptive thresholds impossible. Reducing the period allows for a series
# of RSI values to be computed from the limited warm data, enabling the adaptive logic.
RSI_PERIOD_HIGHER_TF = 2

# Number of higher-timeframe RSI values to use for calculating adaptive percentiles.
# This value must be less than or equal to the number of available valid RSI values.
# With RSI_PERIOD_HIGHER_TF=2 and max 6 higher-TF candles, we can get 4 valid RSI values.
RSI_PERCENTILE_WINDOW = 3 

# Percentiles for adaptive RSI oversold/overbought thresholds
OVERSOLD_PERCENTILE = 10
OVERBOUGHT_PERCENTILE = 90


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


def _calculate_rsi_series(prices: list[float], period: int) -> list[float | None]:
    """Calculates the Relative Strength Index (RSI) for a given price series,
    returning a list of RSI values (None for initial periods where not enough data).
    """
    if len(prices) < period + 1:
        return [None] * len(prices)

    rsi_values = []
    
    # Calculate initial gains and losses for the first 'period' changes
    gains_initial_sum = 0.0
    losses_initial_sum = 0.0
    
    for i in range(1, period + 1):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains_initial_sum += change
        else:
            losses_initial_sum += abs(change)
    
    # Pad with None for initial candles that don't have enough history for RSI
    rsi_values.extend([None] * period)

    avg_gain = gains_initial_sum / period
    avg_loss = losses_initial_sum / period
    
    # Calculate initial RS and RSI
    if avg_loss == 0:
        rs = float('inf')
    else:
        rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values.append(rsi)

    # Apply Wilder's smoothing for subsequent periods to get the latest RSI values
    for i in range(period + 1, len(prices)):
        change = prices[i] - prices[i - 1]
        current_gain = change if change > 0 else 0.0
        current_loss = abs(change) if change < 0 else 0.0

        avg_gain = ((avg_gain * (period - 1)) + current_gain) / period
        avg_loss = ((avg_loss * (period - 1)) + current_loss) / period

        if avg_loss == 0:
            rs = float('inf')
        else:
            rs = avg_gain / rs
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    
    return rsi_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Current timeframe data (hourly candles)
        current_tf_prices = [c.close for c in pair_data.warm]
        
        # Higher timeframe data (e.g., 4-hour candles from hourly data)
        higher_tf_prices = _aggregate_closes_by_factor(pair_data.warm, HIGHER_TIMEFRAME_FACTOR)

        # 2. If insufficient data, return "NO_SIGNAL"
        # Check for sufficient data for ARIMA forecast and volatility calculation
        if (
            len(current_tf_prices) < MIN_CANDLES
            or len(current_tf_prices) < LOOKBACK_VOLATILITY_WINDOW
            or not pair_data.hot # Need hot data for current price and timestamp
        ):
            continue

        # Check for sufficient data for higher timeframe RSI calculation and percentile window
        # We need enough aggregated candles to calculate at least RSI_PERCENTILE_WINDOW valid RSI values.
        # The number of valid RSI values will be len(higher_tf_prices) - RSI_PERIOD_HIGHER_TF.
        # So, we need len(higher_tf_prices) - RSI_PERIOD_HIGHER_TF >= RSI_PERCENTILE_WINDOW
        if len(higher_tf_prices) < RSI_PERIOD_HIGHER_TF + RSI_PERCENTILE_WINDOW:
            continue

        # 3. Calculate ARIMA(1,1,0) forecast for the next period
        phi, intercept, sigma = _fit_arima110(current_tf_prices)

        # Skip if ARIMA model is degenerate
        if sigma == 0 or not math.isfinite(phi) or not math.isfinite(intercept):
            continue

        forecast_price = _forecast(current_tf_prices, phi, intercept, FORECAST_HORIZON)
        
        # 4. Determine current_price
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # 5. Calculate adaptive volatility-based threshold for price deviation.
        recent_prices_for_volatility = current_tf_prices[-LOOKBACK_VOLATILITY_WINDOW:]
        price_std = np.std(recent_prices_for_volatility)
        
        deviation_threshold = VOLATILITY_MULTIPLIER * price_std
        
        # Avoid signals based on zero or near-zero volatility
        if deviation_threshold <= 0:
            continue

        # 6. Calculate higher_timeframe_rsi series
        higher_timeframe_rsi_series = _calculate_rsi_series(higher_tf_prices, RSI_PERIOD_HIGHER_TF)
        
        # Filter out None values to get only valid RSI numbers
        valid_htf_rsi_values = [r for r in higher_timeframe_rsi_series if r is not None]

        # Ensure we have enough valid RSI values for the percentile window
        if len(valid_htf_rsi_values) < RSI_PERCENTILE_WINDOW:
            continue
        
        # The history for percentile calculation is the last RSI_PERCENTILE_WINDOW values
        htf_rsi_history = valid_htf_rsi_values[-RSI_PERCENTILE_WINDOW:]
        current_htf_rsi = valid_htf_rsi_values[-1] # The latest RSI value

        # 4. Calculate adaptive RSI oversold and overbought thresholds based on historical percentiles of HTF RSI.
        adaptive_oversold_threshold = np.percentile(htf_rsi_history, OVERSOLD_PERCENTILE)
        adaptive_overbought_threshold = np.percentile(htf_rsi_history, OVERBOUGHT_PERCENTILE)

        # 7-10. Define signal conditions based on ARIMA forecast and RSI confirmation
        buy_signal_condition_arima = forecast_price > (current_price + deviation_threshold)
        sell_signal_condition_arima = forecast_price < (current_price - deviation_threshold)

        buy_signal_condition_rsi = current_htf_rsi < adaptive_oversold_threshold
        sell_signal_condition_rsi = current_htf_rsi > adaptive_overbought_threshold

        # 11-12. Combine conditions for final signal
        if buy_signal_condition_arima and buy_signal_condition_rsi:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif sell_signal_condition_arima and sell_signal_condition_rsi:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals