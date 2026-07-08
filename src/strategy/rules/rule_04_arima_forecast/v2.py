from __future__ import annotations

import math
import statistics
from datetime import datetime

import numpy as np

from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle


# Minimum warm candles needed for a reliable ARIMA fit
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Signal fires when forecast exceeds current price by > this many forecast std devs
SIGNAL_THRESHOLD = 1.5

# Volatility Filter Parameters
# Number of warm candles for one volatility calculation (e.g., 24 for daily volatility)
VOLATILITY_PERIOD = 24
# Minimum number of past volatility observations to calculate percentiles (e.g., 7 for a week of daily volatilities)
MIN_HISTORICAL_VOLATILITY_OBSERVATIONS = 7
# Lower percentile for acceptable volatility range (e.g., 25th percentile)
VOLATILITY_LOWER_PERCENTILE = 25
# Upper percentile for acceptable volatility range (e.g., 75th percentile)
VOLATILITY_UPPER_PERCENTILE = 75

# Calculate the total minimum warm candles required, considering both ARIMA and volatility filter
MIN_CANDLES_FOR_VOLATILITY_CALC = VOLATILITY_PERIOD + MIN_HISTORICAL_VOLATILITY_OBSERVATIONS
REQUIRED_TOTAL_CANDLES = max(MIN_CANDLES, MIN_CANDLES_FOR_VOLATILITY_CALC)


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


def _calculate_std_dev(prices: list[float]) -> float:
    """Calculates the standard deviation of a list of prices."""
    if len(prices) < 2:
        return 0.0
    return statistics.stdev(prices)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        if len(pair_data.warm) < REQUIRED_TOTAL_CANDLES or not pair_data.hot:
            continue

        warm_close_prices = [c.close for c in pair_data.warm]

        # --- Volatility Filter Logic ---
        # Calculate current volatility (std dev of last N candles)
        current_volatility_prices = warm_close_prices[-VOLATILITY_PERIOD:]
        current_volatility = _calculate_std_dev(current_volatility_prices)

        if current_volatility <= 0:  # Skip if current volatility is zero (flat market)
            continue

        # Calculate historical volatilities for percentile range
        historical_volatilities = []
        # Collect MIN_HISTORICAL_VOLATILITY_OBSERVATIONS rolling volatility values
        # from the data *before* the current_volatility window.
        # The loop iterates backwards to get the most recent historical windows first.
        for i in range(1, MIN_HISTORICAL_VOLATILITY_OBSERVATIONS + 1):
            start_idx = len(warm_close_prices) - VOLATILITY_PERIOD - i
            # Ensure the window is valid and within bounds
            if start_idx < 0:
                break
            window_prices = warm_close_prices[start_idx : start_idx + VOLATILITY_PERIOD]
            vol = _calculate_std_dev(window_prices)
            if vol > 0:  # Only consider non-zero volatilities for percentile calculation
                historical_volatilities.append(vol)

        if len(historical_volatilities) < 2:  # Need at least two points for percentile calculation
            continue

        # Calculate volatility bounds based on historical percentiles
        lower_vol_bound = np.percentile(historical_volatilities, VOLATILITY_LOWER_PERCENTILE)
        upper_vol_bound = np.percentile(historical_volatilities, VOLATILITY_UPPER_PERCENTILE)

        # Check if current volatility is within the optimal range
        is_optimal_volatility = (current_volatility >= lower_vol_bound) and \
                                (current_volatility <= upper_vol_bound)

        if not is_optimal_volatility:
            continue  # Suppress signal if volatility is not optimal

        # --- ARIMA Forecast Logic (only if volatility is optimal) ---
        phi, intercept, sigma = _fit_arima110(warm_close_prices)

        if sigma <= 0 or not math.isfinite(phi):
            continue

        forecast_price = _forecast(warm_close_prices, phi, intercept, FORECAST_HORIZON)
        current_price = pair_data.hot[-1].last_price
        std = _forecast_std(phi, sigma, FORECAST_HORIZON)
        ts = pair_data.hot[-1].polled_at

        if std <= 0:
            continue

        deviation = (forecast_price - current_price) / std
        if deviation > SIGNAL_THRESHOLD:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif deviation < -SIGNAL_THRESHOLD:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals