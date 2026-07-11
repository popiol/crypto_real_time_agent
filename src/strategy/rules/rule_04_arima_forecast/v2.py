from __future__ import annotations

import math
import statistics

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


# Minimum warm candles needed for a reliable fit
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Signal fires when forecast exceeds current price by > this many forecast std devs
SIGNAL_THRESHOLD = 1.5


def _calculate_acf1(series: list[float], mu: float) -> float:
    """Calculate the sample autocorrelation at lag 1."""
    n = len(series)
    if n < 2:
        return 0.0
    
    numerator = sum((series[i] - mu) * (series[i-1] - mu) for i in range(1, n))
    
    # Denominator for ACF(1) is the variance of the series
    denominator = sum((x - mu)**2 for x in series)
    
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _fit_arima011(prices: list[float]) -> tuple[float, float, float]:
    """Fit ARIMA(0,1,1) to a price series using Method of Moments.

    Returns (theta, mu, sigma_residual) where theta is the MA(1) coefficient
    on the differenced series, mu is the intercept, and sigma_residual is
    the standard deviation of the innovations (epsilon).
    """
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    n_diffs = len(diffs)

    # Need at least two differenced values to calculate ACF(1) and variance
    if n_diffs < 2:
        return 0.0, 0.0, 0.0

    mu = statistics.mean(diffs)
    
    # Calculate sample ACF(1)
    r1 = _calculate_acf1(diffs, mu)

    theta = 0.0
    # Solve r1 = theta / (1 + theta^2) for theta
    # This is a quadratic equation: r1 * theta^2 - theta + r1 = 0
    # For a real solution where |theta| <= 1 (invertibility condition),
    # we must have |r1| <= 0.5. Cap r1 slightly below 0.5 to avoid numerical issues.
    if abs(r1) > 0.5 - 1e-9:
        r1 = (0.5 - 1e-9) if r1 > 0 else (-0.5 + 1e-9)

    if r1 != 0:
        discriminant = 1 - 4 * r1 * r1
        if discriminant >= 0:
            sqrt_discriminant = math.sqrt(discriminant)
            # Choose the solution that ensures invertibility (|theta| <= 1).
            # This corresponds to choosing the root with smaller absolute value.
            # If r1 > 0, theta is positive: (1 - sqrt_discriminant) / (2 * r1)
            # If r1 < 0, theta is negative: (1 + sqrt_discriminant) / (2 * r1)
            if r1 > 0:
                theta = (1 - sqrt_discriminant) / (2 * r1)
            else: # r1 < 0
                theta = (1 + sqrt_discriminant) / (2 * r1)
        # If discriminant < 0, it means no real solution, which should not happen
        # if r1 is capped correctly. theta remains 0.0 in this case.
    # If r1 is 0, theta is 0.

    # Calculate sigma_residual
    # For an MA(1) process d_t = mu + epsilon_t + theta * epsilon_{t-1},
    # Var(d_t) = sigma_residual^2 * (1 + theta^2)
    # So, sigma_residual^2 = Var(d_t) / (1 + theta^2)
    var_diffs = statistics.variance(diffs) if n_diffs >= 2 else 0.0
    
    if var_diffs <= 0 or (1 + theta**2) == 0: # Avoid division by zero or negative variance
        sigma_residual = 0.0
    else:
        sigma_residual = math.sqrt(var_diffs / (1 + theta**2))

    return theta, mu, sigma_residual


def _forecast_arima011(prices: list[float], theta: float, mu: float, horizon: int) -> float:
    """Forecast h steps ahead for ARIMA(0,1,1)."""
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    n_diffs = len(diffs)

    if n_diffs == 0:
        # If no differenced values, cannot estimate last_epsilon.
        # Fallback to a simple trend forecast based on mu.
        return prices[-1] + horizon * mu

    # Calculate residuals (epsilon_t) up to the last observed differenced value.
    # epsilon_t = diff_t - mu - theta * epsilon_{t-1}
    # Initialize epsilon_0 (epsilon at index 0 of diffs) assuming epsilon_{-1} = 0
    epsilons = [0.0] * n_diffs
    epsilons[0] = diffs[0] - mu 

    for i in range(1, n_diffs):
        epsilons[i] = diffs[i] - mu - theta * epsilons[i-1]
    
    last_epsilon = epsilons[-1] # This is epsilon_T

    # 1-step ahead forecast for the differenced series: d_T(1) = mu + theta * epsilon_T
    # 1-step ahead forecast for the price: Y_T(1) = Y_T + d_T(1) = Y_T + mu + theta * epsilon_T
    forecast_price_1_step = prices[-1] + mu + theta * last_epsilon

    # For h >= 2, the forecast of the differenced series is simply mu: d_T(h) = mu.
    # The h-step ahead price forecast is Y_T(h) = Y_T(h-1) + d_T(h) = Y_T(h-1) + mu.
    # This leads to Y_T(h) = Y_T(1) + (h-1) * mu for h >= 1.
    if horizon <= 0:
        return prices[-1] # No forecast, return current price
    elif horizon == 1:
        return forecast_price_1_step
    else:
        return forecast_price_1_step + (horizon - 1) * mu


def _forecast_std_arima011(theta: float, sigma_residual: float, horizon: int) -> float:
    """Exact h-step forecast standard deviation for ARIMA(0,1,1).

    The variance of the h-step forecast error for ARIMA(0,1,1) is:
    Var(e_h) = sigma_residual^2 * [1 + (h-1)*(1+theta)^2]
    """
    if horizon <= 0:
        return 0.0
    
    if sigma_residual <= 0:
        return 0.0

    # For horizon = 1, the formula simplifies to sigma_residual^2 * [1 + 0] = sigma_residual^2
    # So, sigma_residual * sqrt(1) = sigma_residual.
    # The term (horizon - 1) handles this correctly.
    variance_factor = 1.0 + (horizon - 1) * (1.0 + theta)**2
    
    if variance_factor < 0: # Should not happen with real theta, but for robustness
        variance_factor = 0.0

    return sigma_residual * math.sqrt(variance_factor)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Need at least MIN_CANDLES for differencing + fitting.
        # An ARIMA(0,1,1) needs at least 3 prices to get 2 differenced values
        # for ACF(1) calculation and fitting. MIN_CANDLES is 10, which is sufficient.
        if len(pair_data.warm) < MIN_CANDLES or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        
        # Ensure enough prices to calculate diffs for fitting.
        # _fit_arima011 needs at least 2 differenced values, which means 3 original prices.
        if len(prices) < 3:
            continue

        theta, mu, sigma_residual = _fit_arima011(prices)

        # Skip if model fitting failed or parameters are invalid
        if sigma_residual == 0 or not math.isfinite(theta) or not math.isfinite(mu):
            continue

        forecast_price = _forecast_arima011(prices, theta, mu, FORECAST_HORIZON)
        current_price = pair_data.hot[-1].last_price
        std = _forecast_std_arima011(theta, sigma_residual, FORECAST_HORIZON)
        ts = pair_data.hot[-1].polled_at

        # Cannot generate a signal if forecast standard deviation is zero or negative
        if std <= 0:
            continue

        deviation = (forecast_price - current_price) / std
        if deviation > SIGNAL_THRESHOLD:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif deviation < -SIGNAL_THRESHOLD:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals