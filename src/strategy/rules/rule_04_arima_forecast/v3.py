from __future__ import annotations

import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle


# Minimum warm candles needed for a reliable fit for ARIMA
MIN_CANDLES_ARIMA = 10

# Forecast horizon in hours for ARIMA
FORECAST_HORIZON = 3

# Signal fires when forecast exceeds current price by > this many forecast std devs
SIGNAL_THRESHOLD = 1.5

# Period for the Exponential Moving Average (EMA)
EMA_PERIOD = 50


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


def _calculate_ema(prices: list[float], period: int) -> float | None:
    """Calculates the Exponential Moving Average (EMA) for a given list of prices."""
    if len(prices) < period:
        return None

    # Calculate initial SMA for the first 'period' prices
    ema = sum(prices[:period]) / period

    # Calculate EMA for the rest of the prices
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices)):
        ema = (prices[i] - ema) * multiplier + ema
    
    return ema


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm candle data for both ARIMA and EMA
        if len(pair_data.warm) < max(MIN_CANDLES_ARIMA, EMA_PERIOD) or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # --- ARIMA Calculation ---
        phi, intercept, sigma = _fit_arima110(prices)

        if sigma == 0 or not math.isfinite(phi):
            continue

        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        std = _forecast_std(phi, sigma, FORECAST_HORIZON)

        if std <= 0:
            continue

        deviation = (forecast_price - current_price) / std

        # --- EMA Calculation ---
        ema_50 = _calculate_ema(prices, EMA_PERIOD)
        if ema_50 is None: # Should not happen if len(prices) check above is sufficient
            continue

        # --- Signal Generation ---
        # Buy signal: ARIMA forecast significantly higher AND price above EMA (uptrend)
        if (deviation > SIGNAL_THRESHOLD) and (current_price > ema_50):
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Sell signal: ARIMA forecast significantly lower AND price below EMA (downtrend)
        elif (deviation < -SIGNAL_THRESHOLD) and (current_price < ema_50):
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals