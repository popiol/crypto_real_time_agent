"""Rule 04 — Time series: ARIMA(1,1,0) price forecast with tunable threshold.

Fits AR(1) on first-differenced hourly close prices from the warm tier
(equivalent to ARIMA(1,1,0)) via OLS.

Buy signal:  Current price is significantly lower than the 3-hour-ahead forecast,
             relative to forecast uncertainty, by more than N standard deviations.
Sell signal: Current price is significantly higher than the 3-hour-ahead forecast,
             relative to forecast uncertainty, by more than N standard deviations.

Forecast variance is computed exactly from the MA(∞) representation of
ARIMA(1,1,0): Var(e_h) = σ² · Σ_{k=0}^{h-1} ψ_k², where ψ_k = Σ_{j=0}^{k} φ^j.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle
from pydantic import BaseModel, Field


# Minimum warm candles needed for a reliable fit
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Tunable parameter N: Signal fires when normalized deviation exceeds N
N_THRESHOLD = 1.5


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


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        if len(pair_data.warm) < MIN_CANDLES or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        
        # Need at least 2 prices to calculate first difference for ARIMA
        if len(prices) < 2:
            continue

        phi, intercept, sigma = _fit_arima110(prices)

        # Skip if model parameters are invalid or residual standard deviation is zero
        if sigma <= 0 or not math.isfinite(phi) or not math.isfinite(intercept):
            continue

        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        current_price = pair_data.hot[-1].last_price
        std = _forecast_std(phi, sigma, FORECAST_HORIZON)
        ts = pair_data.hot[-1].polled_at

        # Skip if forecast standard deviation is zero or non-finite
        if std <= 0 or not math.isfinite(std):
            continue

        # Calculate normalized deviation: (current_price - arima_forecast) / forecast_uncertainty
        # The pseudocode implies: normalized_deviation = (current_price - arima_forecast) / forecast_uncertainty
        # The original rule used (forecast_price - current_price) / std
        # Let's align with the original rule's interpretation for deviation direction
        # Original: deviation > threshold for BUY, deviation < -threshold for SELL
        # This implies: forecast_price > current_price + threshold*std for BUY
        #               forecast_price < current_price - threshold*std for SELL
        # Pseudocode: normalized_deviation = (current_price - arima_forecast) / forecast_uncertainty
        # if normalized_deviation < -N: buy (current price is much lower than forecast)
        # if normalized_deviation > N: sell (current price is much higher than forecast)
        # Let's use the pseudocode's definition of normalized_deviation for clarity and consistency with the idea.
        normalized_deviation = (current_price - forecast_price) / std

        if normalized_deviation < -N_THRESHOLD:
            # Current price is significantly lower than forecast, relative to uncertainty
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif normalized_deviation > N_THRESHOLD:
            # Current price is significantly higher than forecast, relative to uncertainty
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals