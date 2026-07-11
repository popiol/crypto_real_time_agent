"""Rule 04 — Time series: ARIMA(1,1,0) price forecast (Optimized Deviation Threshold).

This rule is an optimized version of rule_04_arima_forecast_v1.
It fits AR(1) on first-differenced hourly close prices from the warm tier
(equivalent to ARIMA(1,1,0)) via OLS.

Buy signal:  3-hour-ahead forecast > current price by > DEVIATION_MULTIPLIER forecast std devs.
Sell signal: 3-hour-ahead forecast < current price by > DEVIATION_MULTIPLIER forecast std devs.

The DEVIATION_MULTIPLIER parameter is intended for optimization to find
the most profitable threshold for signal generation.

Forecast variance is computed exactly from the MA(∞) representation of
ARIMA(1,1,0): Var(e_h) = σ² · Σ_{k=0}^{h-1} ψ_k², where ψ_k = Σ_{j=0}^{k} φ^j.
"""

from __future__ import annotations

import math
import statistics

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


# Minimum warm candles needed for a reliable fit
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Signal fires when forecast exceeds current price by > this many forecast std devs.
# This parameter is intended for optimization.
# Default value set to the original SIGNAL_THRESHOLD of rule_04_arima_forecast_v1.
DEVIATION_MULTIPLIER = 1.5 

# Rule ID for signals
RULE_ID = "rule_04_arima_forecast_v2_optimized_deviation"


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
    if n >= 2: # n is len(x), which is len(residuals)
        sigma = statistics.stdev(residuals)
    elif residuals: # Should only happen if n=1, meaning 1 residual
        sigma = abs(residuals[0])
    else: # n=0, no residuals
        sigma = 0.0

    return phi, intercept, sigma


def _forecast(prices: list[float], phi: float, intercept: float, horizon: int) -> float:
    """Iterate the ARIMA(1,1,0) recursion h steps ahead."""
    # Ensure there are enough prices to calculate the last difference
    if len(prices) < 2:
        return prices[-1] if prices else 0.0 # Return last price or 0 if no prices

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
            psi_k = (1.0 - phi**(k + 1)) / (1.0 - phi)
        variance += psi_k**2
    return sigma * math.sqrt(variance)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm candles for fitting and hot data for current price and timestamp
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

        # Calculate the deviation of current price from forecast in terms of standard deviations
        deviation = (forecast_price - current_price) / std
        
        # Generate buy/sell signals based on the DEVIATION_MULTIPLIER threshold
        if deviation > DEVIATION_MULTIPLIER:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price, rule_id=RULE_ID))
        elif deviation < -DEVIATION_MULTIPLIER:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price, rule_id=RULE_ID))

    return signals