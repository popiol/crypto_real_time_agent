from __future__ import annotations

import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle


# --- Parameters ---
# Minimum warm candles needed for a reliable ARIMA fit
MIN_CANDLES_ARIMA = 10

# Forecast horizon in hours for ARIMA
FORECAST_HORIZON = 3

# Signal fires when current price deviates from forecast by > this many forecast std devs
ARIMA_DEVIATION_THRESHOLD = 2.0

# RSI parameters
RSI_PERIOD = 14
RSI_OVERSOLD_THRESHOLD = 30  # Changed from 25
RSI_OVERBOUGHT_THRESHOLD = 70  # Changed from 75

# EMA parameters
EMA_PERIOD = 20 # New parameter


# --- Helper functions for ARIMA (adapted from rule_04_arima_forecast_v1) ---
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


# --- Helper function for RSI calculation ---
def _calculate_rsi(prices: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI) for the given prices."""
    if len(prices) <= period:  # Need at least period + 1 prices for period changes
        return None

    gains = [0.0] * (len(prices) - 1)
    losses = [0.0] * (len(prices) - 1)

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains[i - 1] = change
        else:
            losses[i - 1] = abs(change)

    # Calculate initial average gain and loss for the first 'period' changes
    current_avg_gain = sum(gains[:period]) / period
    current_avg_loss = sum(losses[:period]) / period

    # If the number of changes exactly matches the period, we have the final RSI
    if len(gains) == period:
        if current_avg_loss == 0:
            return 100.0 if current_avg_gain > 0 else 50.0
        rs = current_avg_gain / current_avg_loss
        return 100 - (100 / (1 + rs))

    # Calculate smoothed average gain and loss for the remaining changes
    for i in range(period, len(gains)):
        current_avg_gain = ((current_avg_gain * (period - 1)) + gains[i]) / period
        current_avg_loss = ((current_avg_loss * (period - 1)) + losses[i]) / period

    if current_avg_loss == 0:
        # If no losses, RSI is 100 (if there were gains) or 50 (if no gains either)
        return 100.0 if current_avg_gain > 0 else 50.0

    rs = current_avg_gain / current_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# --- Helper function for EMA calculation ---
def _calculate_ema(prices: list[float], period: int) -> float | None:
    """Calculates the Exponential Moving Average (EMA) for the given prices."""
    if len(prices) < period:
        return None

    # Calculate initial SMA for the first 'period' prices
    sma = sum(prices[:period]) / period
    ema = sma

    multiplier = 2 / (period + 1)

    # Apply EMA formula for subsequent prices
    for i in range(period, len(prices)):
        ema = (prices[i] - ema) * multiplier + ema

    return ema


# --- Main signal generation function ---
def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    # Determine the minimum number of warm candles needed for all indicators
    min_required_candles = max(MIN_CANDLES_ARIMA, RSI_PERIOD + 1, EMA_PERIOD)

    for pair, pair_data in data.items():
        # Ensure we have enough data for ARIMA, RSI, and EMA calculations
        if len(pair_data.warm) < min_required_candles or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        ts = pair_data.hot[-1].polled_at
        current_price = pair_data.hot[-1].last_price

        # --- ARIMA Calculation ---
        phi, intercept, sigma = _fit_arima110(prices)

        if sigma <= 0 or not math.isfinite(phi):
            continue

        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        std = _forecast_std(phi, sigma, FORECAST_HORIZON)

        if std <= 0:
            continue

        # Deviation defined as (current_price - forecast_price) / std
        # A positive deviation means current price is above forecast.
        # A negative deviation means current price is below forecast.
        arima_deviation = (current_price - forecast_price) / std

        # --- RSI Calculation ---
        rsi = _calculate_rsi(prices, RSI_PERIOD)
        if rsi is None:
            continue

        # --- EMA Calculation ---
        ema = _calculate_ema(prices, EMA_PERIOD)
        if ema is None:
            continue

        # --- Signal Generation ---
        # Buy signal: current price significantly below forecast AND RSI oversold AND current price above EMA
        if (arima_deviation < -ARIMA_DEVIATION_THRESHOLD and
                rsi < RSI_OVERSOLD_THRESHOLD and
                current_price > ema):
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Sell signal: current price significantly above forecast AND RSI overbought AND current price below EMA
        elif (arima_deviation > ARIMA_DEVIATION_THRESHOLD and
                rsi > RSI_OVERBOUGHT_THRESHOLD and
                current_price < ema):
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals