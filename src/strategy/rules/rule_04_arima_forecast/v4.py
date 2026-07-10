from __future__ import annotations

import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# Minimum warm candles needed for a reliable fit for ARIMA
MIN_CANDLES_ARIMA = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Window for calculating historical volatility (in number of warm candles)
# As per pseudocode, e.g., 20-period standard deviation
VOLATILITY_WINDOW = 20

# Multiplier for the volatility to determine the adaptive deviation threshold
# As per pseudocode, e.g., 1.5
DEVIATION_THRESHOLD_MULTIPLIER = 1.5

# Ensure enough data for both ARIMA and volatility calculation
# Volatility calculation needs at least 2 points for stdev, but we use VOLATILITY_WINDOW
# ARIMA needs MIN_CANDLES_ARIMA.
MIN_CANDLES_TOTAL = max(MIN_CANDLES_ARIMA, VOLATILITY_WINDOW)


def _fit_arima110(prices: list[float]) -> tuple[float, float, float]:
    """Fit ARIMA(1,1,0) to a price series via OLS on first differences.

    Returns (phi, intercept, sigma_residual) where phi is the AR(1) coefficient
    on the differenced series and sigma_residual is in price units.
    """
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    x = diffs[:-1]
    y = diffs[1:]
    n = len(x)

    if n < 2:  # Need at least 2 data points for OLS (x and y)
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
    # statistics.stdev requires at least 2 data points.
    # n is len(diffs) - 1. len(diffs) is len(prices) - 1. So n = len(prices) - 2.
    # If len(prices) is MIN_CANDLES_ARIMA=10, then n=8.
    sigma = statistics.stdev(residuals) if n >= 3 else abs(residuals[0]) if residuals else 0.0

    return phi, intercept, sigma


def _forecast(prices: list[float], phi: float, intercept: float, horizon: int) -> float:
    """Iterate the ARIMA(1,1,0) recursion h steps ahead."""
    # Given MIN_CANDLES_TOTAL, len(prices) will be at least 20, so prices[-1] and prices[-2] are safe.
    last_diff = prices[-1] - prices[-2]
    price = prices[-1]
    delta = last_diff
    for _ in range(horizon):
        delta = intercept + phi * delta
        price += delta
    return price


def _calculate_historical_volatility(prices: list[float], window: int) -> float:
    """Calculate the standard deviation of recent close prices.

    Uses the last 'window' prices.
    Assumes len(prices) >= window and window >= 2 due to MIN_CANDLES_TOTAL check.
    """
    data_for_volatility = prices[-window:]
    return statistics.stdev(data_for_volatility)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm candles for both ARIMA and volatility calculation,
        # and at least one hot tick for current price and timestamp.
        if len(pair_data.warm) < MIN_CANDLES_TOTAL or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]

        # 1. Calculate historical volatility
        volatility = _calculate_historical_volatility(prices, VOLATILITY_WINDOW)
        
        # If volatility is zero or negative (should not happen for stdev),
        # the adaptive threshold would be zero, leading to instant signals on any price change.
        # This usually indicates a lack of meaningful price movement, so we skip.
        if volatility <= 0:
            continue

        adaptive_threshold = DEVIATION_THRESHOLD_MULTIPLIER * volatility

        # 2. Train ARIMA model
        phi, intercept, sigma_residual = _fit_arima110(prices)

        # Check for degenerate model fit (e.g., all prices identical, or numerical issues)
        if sigma_residual == 0 or not math.isfinite(phi):
            continue

        # 3. Generate forecast
        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # 4. Generate signals based on adaptive threshold
        if forecast_price > current_price + adaptive_threshold:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif forecast_price < current_price - adaptive_threshold:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals