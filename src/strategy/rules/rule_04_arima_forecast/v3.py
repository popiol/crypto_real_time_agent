from __future__ import annotations

import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle
from pydantic import BaseModel, Field


# Minimum warm candles needed for a reliable fit and ATR calculation.
# ARIMA needs at least 10 candles.
# ATR calculation for SMA_ATR_PERIOD + ATR_PERIOD candles:
# SMA_ATR_PERIOD = 10
# ATR_PERIOD = 14
# Total candles required for ATR = SMA_ATR_PERIOD + ATR_PERIOD = 10 + 14 = 24.
# So, MIN_CANDLES_TOTAL = max(10, 24) = 24.
MIN_CANDLES_TOTAL = 24

# Forecast horizon in hours (from original rule)
FORECAST_HORIZON = 3

# ATR parameters (SMA_ATR_PERIOD adjusted from 50 to 10 due to warm data limit of 24 candles)
ATR_PERIOD = 14
SMA_ATR_PERIOD = 10

# Adaptive Z-score parameters
BASE_Z_SCORE = 2.0
MIN_ADAPTIVE_Z = 1.0
MAX_ADAPTIVE_Z = 3.0


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


def _calculate_atr_series(candles: list[WarmCandle], atr_period: int) -> list[float]:
    """Calculates a series of ATR values from a list of WarmCandle objects.
    Each ATR value is a Simple Moving Average of True Ranges over `atr_period`.
    Requires `atr_period + 1` candles to compute the first ATR value.
    """
    if len(candles) < atr_period + 1:
        return []

    true_ranges = []
    # Calculate True Ranges starting from the second candle (index 1)
    # as TR[i] needs close[i-1]
    for i in range(1, len(candles)):
        current_candle = candles[i]
        prev_close = candles[i-1].close
        tr = max(current_candle.high - current_candle.low,
                 abs(current_candle.high - prev_close),
                 abs(current_candle.low - prev_close))
        true_ranges.append(tr)

    # To calculate `atr_period`-period ATR, we need at least `atr_period` true_ranges.
    if len(true_ranges) < atr_period:
        return []

    atr_values = []
    # Calculate Simple Moving Average of True Ranges
    for i in range(len(true_ranges) - atr_period + 1):
        atr = statistics.mean(true_ranges[i : i + atr_period])
        atr_values.append(atr)

    return atr_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Check for sufficient warm candles and hot data
        if len(pair_data.warm) < MIN_CANDLES_TOTAL or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        
        # 1. Calculate ARIMA forecast and its standard deviation
        phi, intercept, sigma = _fit_arima110(prices)

        # Skip if model parameters are invalid or residual standard deviation is zero
        if sigma <= 0 or not math.isfinite(phi) or not math.isfinite(intercept):
            continue

        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        current_price = pair_data.hot[-1].last_price
        forecast_std_dev = _forecast_std(phi, sigma, FORECAST_HORIZON)
        ts = pair_data.hot[-1].polled_at

        # Skip if forecast standard deviation is zero or non-finite
        if forecast_std_dev <= 0 or not math.isfinite(forecast_std_dev):
            continue

        # 2. Calculate dynamic volatility factor
        historical_atr_series = _calculate_atr_series(pair_data.warm, ATR_PERIOD)

        # Ensure we have enough ATR values to calculate the long-term average
        if len(historical_atr_series) < SMA_ATR_PERIOD:
            continue
        
        # The last value in historical_atr_series is the current_atr (for the last candle in pair_data.warm)
        current_atr = historical_atr_series[-1]
        
        # Calculate long-term average ATR from the last SMA_ATR_PERIOD values
        long_term_avg_atr = statistics.mean(historical_atr_series[-SMA_ATR_PERIOD:])

        # Avoid division by zero or very small numbers
        if long_term_avg_atr == 0:
            volatility_ratio = 1.0 # Default to no adjustment
        else:
            volatility_ratio = current_atr / long_term_avg_atr
            # Clamp volatility ratio to prevent extreme values from creating absurd thresholds
            volatility_ratio = max(0.1, min(volatility_ratio, 10.0))

        # 3. Define adaptive Z-score threshold
        adaptive_z_score = BASE_Z_SCORE * volatility_ratio

        # Ensure adaptive_z_score doesn't become too small or too large
        adaptive_z_score = max(MIN_ADAPTIVE_Z, min(MAX_ADAPTIVE_Z, adaptive_z_score))

        # 4. Calculate deviation from forecast
        # Normalized deviation: (current_price - arima_forecast) / forecast_uncertainty
        deviation = (current_price - forecast_price) / forecast_std_dev

        # 5. Generate signals
        if deviation < -adaptive_z_score:
            # Current price is significantly lower than forecast, relative to uncertainty
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif deviation > adaptive_z_score:
            # Current price is significantly higher than forecast, relative to uncertainty
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals