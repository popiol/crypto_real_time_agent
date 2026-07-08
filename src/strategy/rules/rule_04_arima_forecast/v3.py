from __future__ import annotations

import math
import statistics

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


# Minimum warm candles needed for a reliable fit for ARIMA
MIN_ARIMA_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Signal fires when forecast exceeds current price by > this many forecast std devs
SIGNAL_THRESHOLD = 1.5

# --- New Trend Confirmation Parameters ---
# Period for the Exponential Moving Average used for long-term trend
LONG_EMA_PERIOD = 20  # e.g., 20-hour EMA

# Absolute price change threshold for trend slope.
# A trend_slope greater than this value indicates an uptrend,
# less than negative this value indicates a downtrend.
# Values within [-TREND_SLOPE_THRESHOLD, TREND_SLOPE_THRESHOLD] are considered neutral.
TREND_SLOPE_THRESHOLD = 0.05  # e.g., 5 cents absolute change per hour


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


def _calculate_ema(prices: list[float], period: int) -> list[float]:
    """Calculate Exponential Moving Average for a given period."""
    if len(prices) < period:
        return []

    ema_values = []
    smoothing_factor = 2 / (period + 1)

    # Calculate initial SMA for the first 'period' values
    initial_sma = statistics.mean(prices[:period])
    ema_values.append(initial_sma)

    # Calculate subsequent EMA values
    for i in range(period, len(prices)):
        current_price = prices[i]
        prev_ema = ema_values[-1]
        current_ema = (current_price - prev_ema) * smoothing_factor + prev_ema
        ema_values.append(current_ema)

    return ema_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm candles for both ARIMA and EMA slope calculation.
        # ARIMA needs MIN_ARIMA_CANDLES.
        # EMA needs LONG_EMA_PERIOD to get its first value, and at least 2 EMA values
        # to calculate a slope, which requires LONG_EMA_PERIOD + 1 candles in total.
        min_required_candles = max(MIN_ARIMA_CANDLES, LONG_EMA_PERIOD + 1)

        if len(pair_data.warm) < min_required_candles or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # --- ARIMA Forecast Calculation ---
        phi, intercept, sigma = _fit_arima110(prices)
        if sigma == 0 or not math.isfinite(phi):
            continue

        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        forecast_std_val = _forecast_std(phi, sigma, FORECAST_HORIZON)

        if forecast_std_val <= 0:
            continue

        # --- Trend Confirmation Calculation ---
        ema_long_values = _calculate_ema(prices, LONG_EMA_PERIOD)
        if len(ema_long_values) < 2:
            # This case should be caught by min_required_candles, but serves as a safeguard.
            continue
        
        # Calculate the slope of the long-term EMA
        trend_slope = ema_long_values[-1] - ema_long_values[-2]

        # --- Signal Generation with Adaptive Trend Confirmation ---
        deviation = (forecast_price - current_price) / forecast_std_val

        # Check for Buy signal: ARIMA forecasts significant increase AND (Uptrend OR Neutral trend)
        if deviation > SIGNAL_THRESHOLD:
            if trend_slope > TREND_SLOPE_THRESHOLD or abs(trend_slope) <= TREND_SLOPE_THRESHOLD:
                signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        
        # Check for Sell signal: ARIMA forecasts significant decrease AND (Downtrend OR Neutral trend)
        elif deviation < -SIGNAL_THRESHOLD:
            if trend_slope < -TREND_SLOPE_THRESHOLD or abs(trend_slope) <= TREND_SLOPE_THRESHOLD:
                signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals