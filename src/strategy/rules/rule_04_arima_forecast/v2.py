from __future__ import annotations

import math
import statistics

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle


# Minimum warm candles needed for a reliable fit for ARIMA
MIN_CANDLES_FOR_ARIMA = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Base K multiplier for the adaptive threshold (replaces fixed SIGNAL_THRESHOLD)
K_BASE = 1.5

# Window size for calculating recent volatility (in hours/candles)
# E.g., 24 for 1 day of hourly data.
VOLATILITY_WINDOW_SHORT = 24

# Window size for calculating long-term average volatility (in hours/candles)
# E.g., 7 * 24 for 1 week of hourly data.
VOLATILITY_WINDOW_LONG = 7 * 24

# Factor to control how sensitive the K multiplier is to volatility changes.
# A higher factor means the threshold adapts more aggressively to recent volatility.
SENSITIVITY_FACTOR = 0.5

# Minimum and maximum bounds for the adaptive K multiplier.
# This prevents the threshold from becoming too small (too many false positives)
# or too large (missing too many signals).
MIN_K_MULTIPLIER = 0.5
MAX_K_MULTIPLIER = 3.0

# Minimum candles required for volatility calculation.
# We need N+1 prices to get N returns. So, for a window of size W, we need W+1 prices.
MIN_CANDLES_FOR_VOLATILITY = VOLATILITY_WINDOW_LONG + 1


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
    # statistics.stdev requires at least 2 data points for a non-zero std.
    # If only one residual, its std dev is typically considered 0.
    sigma = statistics.stdev(residuals) if len(residuals) >= 2 else abs(residuals[0]) if residuals else 0.0

    return phi, intercept, sigma


def _forecast(prices: list[float], phi: float, intercept: float, horizon: int) -> float:
    """Iterate the ARIMA(1,1,0) recursion h steps ahead."""
    if len(prices) < 2:
        return prices[-1] if prices else 0.0 # Cannot forecast without at least 2 prices for initial diff

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


def _calculate_log_returns(prices: list[float]) -> list[float]:
    """Calculates log returns from a list of prices."""
    if len(prices) < 2:
        return []
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0: # Avoid division by zero
            returns.append(math.log(prices[i] / prices[i-1]))
        else:
            returns.append(0.0) # If previous price is zero, return is undefined, treat as 0.0
    return returns


def _calculate_std_dev_of_returns(returns: list[float]) -> float:
    """Calculates the standard deviation of a list of returns.
    Returns 0.0 if there are fewer than 2 returns (std dev is undefined or 0).
    """
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm candles for both ARIMA fitting and volatility calculations.
        min_required_candles = max(MIN_CANDLES_FOR_ARIMA, MIN_CANDLES_FOR_VOLATILITY)
        if len(pair_data.warm) < min_required_candles or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # 1. Fit ARIMA model
        phi, intercept, sigma = _fit_arima110(prices)

        if sigma == 0 or not math.isfinite(phi):
            continue

        # 2. Generate ARIMA forecast and its standard deviation
        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        forecast_std_dev = _forecast_std(phi, sigma, FORECAST_HORIZON)

        if forecast_std_dev <= 0:
            # If forecast_std_dev is zero or negative, the forecast is perfectly certain
            # or invalid, preventing meaningful signal generation.
            continue

        # 3. Calculate adaptive K multiplier based on volatility
        # We need N+1 prices for N returns. The initial check ensures enough data.

        # Calculate recent volatility from the most recent `VOLATILITY_WINDOW_SHORT` returns
        recent_prices_for_volatility = prices[-(VOLATILITY_WINDOW_SHORT + 1):]
        recent_returns = _calculate_log_returns(recent_prices_for_volatility)
        recent_volatility = _calculate_std_dev_of_returns(recent_returns)

        # Calculate long-term average volatility from the most recent `VOLATILITY_WINDOW_LONG` returns
        long_term_prices_for_volatility = prices[-(VOLATILITY_WINDOW_LONG + 1):]
        long_term_returns = _calculate_log_returns(long_term_prices_for_volatility)
        long_term_avg_volatility = _calculate_std_dev_of_returns(long_term_returns)

        adaptive_k_multiplier = K_BASE
        if long_term_avg_volatility > 0:
            # Calculate adjustment factor: positive if recent volatility > long-term avg, negative otherwise
            adjustment_factor = (recent_volatility - long_term_avg_volatility) / long_term_avg_volatility
            adaptive_k_multiplier = K_BASE * (1 + adjustment_factor * SENSITIVITY_FACTOR)
            
            # Clip the adaptive_k_multiplier within defined bounds
            adaptive_k_multiplier = max(MIN_K_MULTIPLIER, min(MAX_K_MULTIPLIER, adaptive_k_multiplier))
        # If long_term_avg_volatility is 0, the adaptive_k_multiplier remains at K_BASE,
        # indicating no volatility-based adjustment is possible.

        # 4. Calculate the adaptive threshold
        adaptive_threshold_value = adaptive_k_multiplier * forecast_std_dev

        # 5. Generate signals based on the adaptive threshold
        deviation = forecast_price - current_price

        if deviation > adaptive_threshold_value:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif deviation < -adaptive_threshold_value:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals