from __future__ import annotations

import math
import statistics
import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


# Minimum warm candles needed for a reliable fit
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Lookback window for calculating recent volatility (in hours/candles)
LOOKBACK_VOLATILITY_WINDOW = 24

# Multiplier for the historical standard deviation to determine the deviation threshold
VOLATILITY_MULTIPLIER = 1.5


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
        if len(pair_data.warm) < MIN_CANDLES or not pair_data.hot:
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
        
        # Use numpy for standard deviation for consistency with pseudocode and efficiency
        price_std = np.std(recent_prices_for_volatility)
        
        # If volatility is zero or near-zero, the threshold would be zero, making any
        # tiny deviation a signal. We might want to avoid this or set a minimum.
        # For now, if price_std is 0, deviation_threshold will be 0.
        # This means any price_diff > 0 or < 0 will trigger.
        deviation_threshold = VOLATILITY_MULTIPLIER * price_std
        
        if deviation_threshold <= 0: # Avoid division by zero or overly sensitive threshold
            continue

        # 3. Compare forecast with current price using adaptive threshold
        price_diff = forecast_price - current_price
        
        if price_diff > deviation_threshold:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif price_diff < -deviation_threshold:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals