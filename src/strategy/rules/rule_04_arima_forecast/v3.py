"""Rule 51b1126b-9e9e-4b52-8f29-8dbbe7d3a5c7 — ARIMA Forecast Deviation with Adaptive Volatility Threshold."""
from __future__ import annotations

import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# Minimum warm candles needed for a reliable ARIMA fit
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Period for Average True Range (ATR) calculation (in warm candles/hours)
ATR_PERIOD = 14

# Base deviation multiplier (unitless, similar to original SIGNAL_THRESHOLD)
K_BASE = 1.5

# Volatility adjustment factor (units of 1/price)
# This factor scales the ATR, which is in price units, to become unitless
# so it can be added to K_BASE.
V_ADJ = 0.1


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


def _calculate_atr(candles: list[WarmCandle], period: int) -> float:
    """Calculates the Average True Range (ATR) for a given period."""
    if len(candles) <= period:  # Need at least period + 1 candles for period TRs
        return 0.0

    true_ranges = []
    # Start from the second candle to calculate TR, as it needs prev_close
    for i in range(1, len(candles)):
        high_i = candles[i].high
        low_i = candles[i].low
        close_prev = candles[i-1].close

        tr = max(
            high_i - low_i,
            abs(high_i - close_prev),
            abs(low_i - close_prev)
        )
        true_ranges.append(tr)
    
    # Take the last 'period' true ranges
    if len(true_ranges) < period:
        return 0.0 # Not enough TRs to average

    return statistics.mean(true_ranges[-period:])


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        if len(pair_data.warm) < MIN_CANDLES or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        phi, intercept, sigma = _fit_arima110(prices)

        if sigma == 0 or not math.isfinite(phi):
            continue

        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        current_price = pair_data.hot[-1].last_price
        forecast_std_err = _forecast_std(phi, sigma, FORECAST_HORIZON)
        ts = pair_data.hot[-1].polled_at

        if forecast_std_err <= 0:
            continue

        # Calculate current volatility using ATR
        current_volatility = _calculate_atr(pair_data.warm, ATR_PERIOD)

        # Calculate the dynamic signal threshold
        # If current_volatility is 0 (e.g., not enough data), this term becomes 0,
        # and the threshold reverts to a static K_BASE * forecast_std_err.
        signal_threshold_multiplier = K_BASE + (V_ADJ * current_volatility)
        
        # Ensure the multiplier is not negative, though unlikely with positive K_BASE and V_ADJ
        signal_threshold_multiplier = max(0.0, signal_threshold_multiplier)

        # The actual threshold in price units
        dynamic_signal_threshold = signal_threshold_multiplier * forecast_std_err

        # Generate signals based on the dynamic threshold
        if current_price < forecast_price - dynamic_signal_threshold:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif current_price > forecast_price + dynamic_signal_threshold:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals