from __future__ import annotations

import math
import statistics
import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


# Minimum warm candles needed for a reliable ARIMA(1,1,0) fit (at least 3, 10 is better)
MIN_CANDLES_FOR_ARIMA_FIT = 10

# Forecast horizon in hours (how many future hours to forecast)
FORECAST_HORIZON = 3

# Lookback window for calculating recent volatility (in hours/candles)
LOOKBACK_VOLATILITY_WINDOW = 24

# Multiplier for the historical standard deviation to determine the adaptive deviation threshold
VOLATILITY_MULTIPLIER = 1.5

# New parameter: Period for the Exponential Moving Average on the higher timeframe (hourly candles)
HIGHER_TF_EMA_PERIOD = 20 # E.g., a 20-hour EMA for trend confirmation

# Overall minimum warm candles required for all calculations to be valid.
# This ensures enough data for ARIMA fitting, volatility calculation, and EMA calculation.
MIN_WARM_CANDLES_REQUIRED = max(
    MIN_CANDLES_FOR_ARIMA_FIT,
    LOOKBACK_VOLATILITY_WINDOW,
    HIGHER_TF_EMA_PERIOD
)


def _fit_arima110(prices: list[float]) -> tuple[float, float, float]:
    """Fit ARIMA(1,1,0) to a price series via OLS on first differences.

    Returns (phi, intercept, sigma_residual) where phi is the AR(1) coefficient
    on the differenced series and sigma_residual is in price units.
    """
    # Calculate first differences
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    
    # For AR(1) model on differences: diff_t = intercept + phi * diff_{t-1} + e_t
    # Here, x corresponds to diff_{t-1} and y corresponds to diff_t
    x = diffs[:-1]
    y = diffs[1:]
    n = len(x)

    # Need at least two data points (diffs) to fit an AR(1) model
    if n < 2:
        return 0.0, 0.0, 0.0

    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)

    cov_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    var_x = sum((xi - x_mean) ** 2 for xi in x)

    if var_x == 0: # Avoid division by zero if all x values are the same
        phi = 0.0
        intercept = y_mean
    else:
        phi = cov_xy / var_x
        intercept = y_mean - phi * x_mean

    # Calculate residuals and their standard deviation (sigma)
    residuals = [yi - (phi * xi + intercept) for xi, yi in zip(x, y)]
    # Use sample standard deviation (N-1 degrees of freedom), statistics.stdev does this by default
    sigma = statistics.stdev(residuals) if n >= 2 else abs(residuals[0]) if residuals else 0.0

    return phi, intercept, sigma


def _forecast(prices: list[float], phi: float, intercept: float, horizon: int) -> float:
    """Iterate the ARIMA(1,1,0) recursion h steps ahead to get a price forecast."""
    if len(prices) < 2: # Need at least two prices to calculate the last difference
        return prices[-1] if prices else 0.0
        
    last_diff = prices[-1] - prices[-2] # The last known difference
    price = prices[-1] # The current price
    delta = last_diff # The forecast for the next difference
    
    # Iterate the ARIMA(1,1,0) model forward for 'horizon' steps
    for _ in range(horizon):
        delta = intercept + phi * delta # Forecast next difference based on AR(1) model
        price += delta # Add the forecasted difference to the price
    return price


def _forecast_std(phi: float, sigma: float, horizon: int) -> float:
    """Calculates the exact h-step forecast standard deviation for ARIMA(1,1,0).

    This function calculates the standard deviation of the forecast error,
    which can be used to construct prediction intervals.
    """
    variance = 0.0
    for k in range(horizon):
        # ψ_k are the impulse-response weights
        psi_k = sum(phi**j for j in range(k + 1))
        variance += psi_k**2
    return sigma * math.sqrt(variance)


def _calculate_ema(prices: list[float], period: int) -> float | None:
    """Calculates the Exponential Moving Average (EMA) for the last price in the series.
    Returns None if not enough data is available for the given period.
    """
    if len(prices) < period:
        return None

    # Calculate initial Simple Moving Average (SMA) for the first 'period' prices
    ema = sum(prices[:period]) / period
    multiplier = 2 / (period + 1)

    # Apply the EMA formula for the rest of the prices
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have current tick data for the current price and timestamp
        if not pair_data.hot:
            continue
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # Extract close prices from warm (hourly) candles. This is our higher timeframe data.
        warm_prices = [c.close for c in pair_data.warm]
        
        # Ensure enough warm candles for all calculations (ARIMA, volatility, and EMA)
        if len(warm_prices) < MIN_WARM_CANDLES_REQUIRED:
            continue

        # 1. Calculate ARIMA(1,1,0) forecast using warm (hourly) prices
        phi, intercept, sigma = _fit_arima110(warm_prices)

        # Skip if ARIMA model is degenerate (e.g., no variance in residuals or non-finite phi)
        if sigma == 0 or not math.isfinite(phi):
            continue

        forecast_price = _forecast(warm_prices, phi, intercept, FORECAST_HORIZON)

        # 2. Calculate adaptive deviation threshold based on recent volatility
        # Use a slice of warm_prices for volatility calculation over the LOOKBACK_VOLATILITY_WINDOW
        recent_prices_for_volatility = warm_prices[-LOOKBACK_VOLATILITY_WINDOW:]
        
        # Use numpy for standard deviation for robustness and efficiency
        price_std = np.std(recent_prices_for_volatility)
        
        deviation_threshold = VOLATILITY_MULTIPLIER * price_std
        
        # If volatility is zero or near-zero, the threshold would be zero, making any
        # tiny deviation a signal. We prevent this to avoid overly sensitive signals.
        if deviation_threshold <= 0:
            continue

        # 3. Calculate Higher Timeframe EMA for trend confirmation
        higher_tf_current_price = warm_prices[-1] # The latest close price on the higher timeframe (hourly)
        higher_tf_ema = _calculate_ema(warm_prices, HIGHER_TF_EMA_PERIOD)

        # This check acts as a safeguard; MIN_WARM_CANDLES_REQUIRED should prevent higher_tf_ema from being None
        if higher_tf_ema is None:
            continue

        # 4. Apply Multi-Timeframe Trend Confirmation Logic
        price_diff = forecast_price - current_price
        
        # Buy Signal: ARIMA forecasts adaptively higher AND higher timeframe price is above EMA
        if (price_diff > deviation_threshold) and \
           (higher_tf_current_price > higher_tf_ema):
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Sell Signal: ARIMA forecasts adaptively lower AND higher timeframe price is below EMA
        elif (price_diff < -deviation_threshold) and \
             (higher_tf_current_price < higher_tf_ema):
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals