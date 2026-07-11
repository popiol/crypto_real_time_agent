from __future__ import annotations

import math
import statistics
from datetime import datetime

# Assuming these models are available in the execution environment
from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle, ColdMonth


# --- Rule Constants ---
# Minimum warm candles needed for a reliable ARIMA fit
MIN_CANDLES_FOR_ARIMA = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Standard deviation multiplier for signal generation
DEVIATION_MULTIPLIER = 2.0

# RSI parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Combined minimum candles needed for both ARIMA and RSI calculation
# RSI needs at least RSI_PERIOD + 1 candles for its first calculation.
# ARIMA needs MIN_CANDLES_FOR_ARIMA.
MIN_TOTAL_CANDLES = max(MIN_CANDLES_FOR_ARIMA, RSI_PERIOD + 1)


# --- ARIMA Helper Functions (copied from original rule_04_arima_forecast_v1) ---

def _fit_arima110(prices: list[float]) -> tuple[float, float, float]:
    """Fit ARIMA(1,1,0) to a price series via OLS on first differences.

    Returns (phi, intercept, sigma_residual) where phi is the AR(1) coefficient
    on the differenced series and sigma_residual is in price units.
    """
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    x = diffs[:-1]
    y = diffs[1:]
    n = len(x)

    if n < 2:  # Need at least two differences to fit AR(1), which means 3 prices
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
    # This function is called after _fit_arima110 which ensures len(prices) >= 3
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


# --- RSI Helper Function ---

def _calculate_rsi(prices: list[float], period: int) -> float | None:
    """
    Calculates the Relative Strength Index (RSI) for a given list of prices.
    Returns the latest RSI value or None if not enough data.
    """
    if len(prices) < period + 1:
        return None

    gains = []
    losses = []
    # Calculate price changes (gains and losses)
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))

    # Ensure we have enough changes to calculate the initial average
    if len(gains) < period:
        return None

    # Calculate initial average gain and loss (SMA for the first 'period' changes)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Calculate subsequent averages using EMA-like smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Handle edge cases for RS and RSI calculation
    if avg_loss == 0:
        if avg_gain == 0:
            # No movement at all, or perfectly flat. RSI is conventionally 50.
            return 50.0
        # Only gains, no losses. RS is infinite, RSI is 100.
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have current hot data for price and timestamp
        if not pair_data.hot:
            continue

        # Ensure enough warm candles for both ARIMA and RSI calculations
        if len(pair_data.warm) < MIN_TOTAL_CANDLES:
            continue

        prices = [c.close for c in pair_data.warm]
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # --- ARIMA Forecast Calculation ---
        # MIN_TOTAL_CANDLES (15) ensures len(prices) is sufficient for ARIMA (min 3)
        phi, intercept, sigma = _fit_arima110(prices)

        # Skip if ARIMA parameters are unreliable
        if sigma <= 0 or not math.isfinite(phi):
            continue

        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        forecast_std_dev = _forecast_std(phi, sigma, FORECAST_HORIZON)

        # Skip if forecast standard deviation is unreliable
        if forecast_std_dev <= 0:
            continue

        # --- RSI Confirmation Calculation ---
        current_rsi = _calculate_rsi(prices, RSI_PERIOD)
        if current_rsi is None:
            # This check is mostly redundant if MIN_TOTAL_CANDLES is correctly set,
            # but acts as a safeguard against unexpected RSI calculation failures.
            continue

        # --- Signal Generation with Confirmation ---
        buy_threshold = forecast_price - (DEVIATION_MULTIPLIER * forecast_std_dev)
        sell_threshold = forecast_price + (DEVIATION_MULTIPLIER * forecast_std_dev)

        # Buy signal: current price significantly below forecast AND RSI indicates oversold
        if current_price < buy_threshold and current_rsi <= RSI_OVERSOLD:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Sell signal: current price significantly above forecast AND RSI indicates overbought
        elif current_price > sell_threshold and current_rsi >= RSI_OVERBOUGHT:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals