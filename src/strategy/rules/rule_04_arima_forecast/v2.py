from __future__ import annotations

import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# Minimum warm candles needed for a reliable fit
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# ARIMA model parameters
ARIMA_ORDER = (1, 1, 0)
ARIMA_LOOKBACK = 60  # bars for ARIMA model training

# Volatility window parameters
VOLATILITY_WINDOW_SHORT = 20  # bars for recent volatility
VOLATILITY_WINDOW_LONG = 100  # bars for long-term average volatility

# Baseline deviation multiplier
BASE_K = 2.0


def _fit_arima110(prices: list[float]) -> tuple[float, float, float]:
    """Fit ARIMA(1,1,0) to a price series via OLS on first differences.

    Returns (phi, intercept, sigma_residual) where phi is the AR(1) coefficient
    on the differenced series and sigma_residual is in price units.
    """
    if len(prices) < 2:  # Need at least two prices to calculate a difference
        return 0.0, 0.0, 0.0

    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    
    if len(diffs) < 2: # Need at least two differences for OLS (x,y pairs)
        # If only one diff, we can't fit AR(1). Assume no AR component,
        # and forecast is just the mean of the diff (or 0 if no diffs).
        # For simplicity, if only one diff, phi=0, intercept=that diff, sigma=0.
        if len(diffs) == 1:
            return 0.0, diffs[0], 0.0
        return 0.0, 0.0, 0.0


    x = diffs[:-1]
    y = diffs[1:]
    n = len(x)

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
        return prices[-1] if prices else 0.0 # Cannot forecast without prior differences

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


def _safe_stdev(data: list[float]) -> float:
    """Calculates standard deviation, returning 0.0 if data is insufficient or constant."""
    if len(data) < 2:
        return 0.0
    try:
        return statistics.stdev(data)
    except statistics.StatisticsError:
        # This typically happens if all data points are identical.
        return 0.0


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        if not pair_data.hot:
            continue

        prices_warm = [c.close for c in pair_data.warm]
        num_warm_candles = len(prices_warm)

        # Ensure enough data for overall processing
        if num_warm_candles < MIN_CANDLES:
            continue

        # --- Calculate adaptive K-factor ---
        # Determine actual window sizes based on available warm data
        volatility_short_window_actual = min(VOLATILITY_WINDOW_SHORT, num_warm_candles)
        volatility_long_window_actual = min(VOLATILITY_WINDOW_LONG, num_warm_candles)

        adaptive_k = BASE_K # Default fallback

        # Only calculate adaptive K if enough data for both short and long windows
        if volatility_short_window_actual >= 2 and volatility_long_window_actual >= 2:
            recent_prices_volatility = prices_warm[-volatility_short_window_actual:]
            long_term_prices_volatility = prices_warm[-volatility_long_window_actual:]

            current_volatility = _safe_stdev(recent_prices_volatility)
            avg_volatility = _safe_stdev(long_term_prices_volatility)

            if avg_volatility > 0:
                # Calculate adaptive K-factor
                adaptive_k = BASE_K * (1 + (current_volatility - avg_volatility) / avg_volatility)
                # Ensure adaptive_k doesn't go below 0 or become excessively large;
                # a simple floor/cap could be added if needed, but for now, follow pseudocode.
                if adaptive_k < 0.1: # Prevent extremely small K, which can lead to excessive signals
                    adaptive_k = 0.1
            # else: adaptive_k remains BASE_K (fallback)

        # --- Train and forecast with ARIMA model ---
        arima_lookback_actual = min(ARIMA_LOOKBACK, num_warm_candles)

        # ARIMA model requires at least 2 prices to calculate differences for fitting
        # and MIN_CANDLES for a reliable fit.
        if arima_lookback_actual < MIN_CANDLES:
            continue
        
        historical_prices_arima = prices_warm[-arima_lookback_actual:]

        phi, intercept, sigma = _fit_arima110(historical_prices_arima)

        if sigma <= 0 or not math.isfinite(phi):
            continue

        forecast_price = _forecast(historical_prices_arima, phi, intercept, FORECAST_HORIZON)
        current_price = pair_data.hot[-1].last_price
        forecast_std_dev = _forecast_std(phi, sigma, FORECAST_HORIZON)
        ts = pair_data.hot[-1].polled_at

        if forecast_std_dev <= 0:
            continue

        # --- Generate signal ---
        normalized_deviation = (forecast_price - current_price) / forecast_std_dev

        if normalized_deviation > adaptive_k:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif normalized_deviation < -adaptive_k:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals