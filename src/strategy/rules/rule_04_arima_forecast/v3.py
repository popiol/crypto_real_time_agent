"""Rule 04 — Time series: ARIMA(1,1,0) price forecast (Adaptive Threshold Multiplier).

This rule modifies rule_04_arima_forecast_v2 to dynamically adjust the multiplier (K)
applied to the ARIMA forecast's standard deviation when determining the signal threshold.
Instead of a fixed K, the multiplier adapts based on recent market volatility.
Specifically, a higher market volatility will lead to a higher K, requiring a larger
price deviation from the forecast to generate a signal, thereby filtering out noise
in choppy markets.

Buy signal: current price < forecast_price - (adaptive_k * forecast_std_dev)
Sell signal: current price > forecast_price + (adaptive_k * forecast_std_dev)
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle


# Minimum warm candles needed for a reliable ARIMA fit
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Base K multiplier for the forecast standard deviation
BASE_K_MULTIPLIER = 2.0

# Lookback window for calculating current market volatility (in number of hourly candles)
VOLATILITY_LOOKBACK_WINDOW = 30

# Lookback window for calculating long-term average volatility (in number of hourly candles)
LONG_TERM_AVG_VOLATILITY_WINDOW = 100

# Rule ID for signals
RULE_ID = "rule_04_arima_forecast_v2_adaptive_k"


def _fit_arima110(prices: list[float]) -> tuple[float, float, float]:
    """Fit ARIMA(1,1,0) to a price series via OLS on first differences.

    Returns (phi, intercept, sigma_residual) where phi is the AR(1) coefficient
    on the differenced series and sigma_residual is in price units.
    """
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    x = diffs[:-1]
    y = diffs[1:]
    n = len(x)

    if n < 2:  # Need at least 2 data points for OLS, implies len(prices) >= 3
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

    # Calculate standard deviation of residuals.
    # statistics.stdev requires at least 2 data points.
    if n >= 2:  # n is len(x), which is len(residuals)
        sigma = statistics.stdev(residuals)
    elif residuals:  # Should only happen if n=1, meaning 1 residual
        sigma = abs(residuals[0])
    else:  # n=0, no residuals
        sigma = 0.0

    return phi, intercept, sigma


def _forecast(prices: list[float], phi: float, intercept: float, horizon: int) -> float:
    """Iterate the ARIMA(1,1,0) recursion h steps ahead."""
    # Ensure there are enough prices to calculate the last difference
    if len(prices) < 2:
        return prices[-1] if prices else 0.0  # Return last price or 0 if no prices

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
        # Calculate psi_k = 1 + phi + phi^2 + ... + phi^k
        # This is a geometric series sum: (1 - phi^(k+1)) / (1 - phi)
        # Handle phi=1 separately to avoid division by zero
        if phi == 1.0:
            psi_k = k + 1
        else:
            psi_k = (1.0 - phi ** (k + 1)) / (1.0 - phi)
        variance += psi_k**2
    return sigma * math.sqrt(variance)


def _calculate_volatility(prices_slice: list[float], window: int) -> float:
    """Calculates the standard deviation of log returns for a given window."""
    # We need `window` returns, which means `window + 1` prices.
    # `prices_slice` should already contain the correct number of prices.
    if len(prices_slice) < 2:  # Need at least 2 prices for 1 return
        return 0.0

    log_returns = []
    for i in range(1, len(prices_slice)):
        # Ensure prices are positive before taking log
        if prices_slice[i - 1] > 0 and prices_slice[i] > 0:
            log_returns.append(math.log(prices_slice[i] / prices_slice[i - 1]))
        else:
            # If any price is non-positive, volatility cannot be calculated meaningfully
            return 0.0

    if len(log_returns) < 2:
        # stdev requires at least 2 data points; if only one return, stdev is 0.
        # If all returns are identical, stdev will be 0.
        return 0.0 # statistics.stdev would raise StatisticsError for len(log_returns)=1

    return statistics.stdev(log_returns)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm candles for ARIMA fitting.
        # Also, ensure enough data for long-term volatility calculation if it's needed.
        # The adaptive_k logic will handle cases where there aren't enough candles
        # for the full volatility windows by defaulting to BASE_K_MULTIPLIER.
        if len(pair_data.warm) < MIN_CANDLES or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]

        # Fit ARIMA(1,1,0) model to the historical prices
        phi, intercept, sigma = _fit_arima110(prices)

        # Skip if the model parameters are invalid or sigma is zero
        if sigma <= 0 or not math.isfinite(phi) or not math.isfinite(intercept):
            continue

        # Generate forecast and calculate its standard deviation
        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        current_price = pair_data.hot[-1].last_price
        std = _forecast_std(phi, sigma, FORECAST_HORIZON)
        ts = pair_data.hot[-1].polled_at

        # If forecast standard deviation is non-positive, we cannot determine deviation reliably
        if std <= 0:
            continue

        # --- Adaptive K multiplier calculation ---
        adaptive_k = BASE_K_MULTIPLIER  # Default value

        # Check if enough historical prices are available for volatility calculation
        # We need `LONG_TERM_AVG_VOLATILITY_WINDOW + 1` prices to calculate
        # `LONG_TERM_AVG_VOLATILITY_WINDOW` log returns for the long-term average.
        if len(prices) >= LONG_TERM_AVG_VOLATILITY_WINDOW + 1:
            # Get slices of prices for current and long-term volatility
            # The slices are `window_size + 1` elements long to calculate `window_size` returns.
            current_volatility_prices = prices[-VOLATILITY_LOOKBACK_WINDOW - 1 :]
            long_term_volatility_prices = prices[-LONG_TERM_AVG_VOLATILITY_WINDOW - 1 :]

            current_market_volatility = _calculate_volatility(
                current_volatility_prices, VOLATILITY_LOOKBACK_WINDOW
            )
            long_term_avg_volatility = _calculate_volatility(
                long_term_volatility_prices, LONG_TERM_AVG_VOLATILITY_WINDOW
            )

            if long_term_avg_volatility == 0:
                # If long-term volatility is zero, treat ratio as 1.0 to avoid division by zero
                # and prevent K from becoming excessively small.
                volatility_ratio = 1.0
            else:
                volatility_ratio = current_market_volatility / long_term_avg_volatility

            # Apply floor of 1.0 to adaptive_k to ensure a minimum threshold is always applied.
            # This prevents K from becoming too small in extremely low volatility periods,
            # which could lead to excessive signals from minor deviations.
            adaptive_k = max(BASE_K_MULTIPLIER * volatility_ratio, 1.0)
        else:
            # Not enough data for volatility calculation, use base K
            adaptive_k = BASE_K_MULTIPLIER

        # Calculate the price deviation from the forecast
        price_deviation = current_price - forecast_price
        threshold = adaptive_k * std

        # Generate buy/sell signals based on the adaptive threshold
        if price_deviation < -threshold:
            # Current price is significantly below forecast, indicating undervaluation -> Buy
            signals.append(
                BuySignal(pair=pair, timestamp=ts, price=current_price, rule_id=RULE_ID)
            )
        elif price_deviation > threshold:
            # Current price is significantly above forecast, indicating overvaluation -> Sell
            signals.append(
                SellSignal(pair=pair, timestamp=ts, price=current_price, rule_id=RULE_ID)
            )

    return signals