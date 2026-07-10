"""ARIMA Forecast with RSI Confirmation."""
from __future__ import annotations

import math
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# Minimum warm candles needed for a reliable fit for ARIMA
MIN_CANDLES_ARIMA = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Signal fires when forecast exceeds current price by > this percentage deviation
FORECAST_DEVIATION_THRESHOLD = 0.005 # e.g., 0.5% deviation

# RSI parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Minimum warm candles needed for RSI calculation
MIN_CANDLES_RSI = RSI_PERIOD + 1 # At least (period + 1) candles to get one RSI value


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
    if len(prices) < 2: # Need at least 2 prices to get a last_diff
        return prices[-1] if prices else 0.0 # Return last price if not enough data for diff

    last_diff = prices[-1] - prices[-2]
    price = prices[-1]
    delta = last_diff
    for _ in range(horizon):
        delta = intercept + phi * delta
        price += delta
    return price


def _calculate_rsi(prices: list[float], period: int) -> float:
    """Calculates the Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return 0.0 # Not enough data for RSI calculation

    gains = []
    losses = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))

    # Calculate initial average gain and loss over the first 'period' changes
    # 'gains' and 'losses' lists have len(prices) - 1 elements.
    # The initial check `len(prices) < period + 1` ensures `len(gains) >= period`.
    avg_gain = sum(gains[0:period]) / period
    avg_loss = sum(losses[0:period]) / period

    # Calculate subsequent average gain and loss using Wilder's smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0 # If no losses, RSI is 100 (or 50 if no changes)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    # Determine the minimum number of warm candles required for both calculations
    min_required_candles = max(MIN_CANDLES_ARIMA, MIN_CANDLES_RSI)

    for pair, pair_data in data.items():
        if len(pair_data.warm) < min_required_candles or not pair_data.hot:
            continue

        # Extract closing prices from warm candles
        prices = [c.close for c in pair_data.warm]
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # --- ARIMA Forecast ---
        phi, intercept, sigma = _fit_arima110(prices)

        # Check for invalid ARIMA fit parameters.
        # Insufficient data for ARIMA is broadly handled by `min_required_candles`,
        # but specific fit issues (e.g., zero variance) are checked here.
        if sigma == 0 or not math.isfinite(phi):
            continue

        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        
        # Calculate percentage deviation
        if current_price == 0: # Avoid division by zero for deviation calculation
            continue
        price_deviation = (forecast_price - current_price) / current_price

        # --- RSI Confirmation ---
        current_rsi = _calculate_rsi(prices, RSI_PERIOD)
        # If _calculate_rsi returns 0.0 because of insufficient data, skip this pair.
        # Otherwise, 0.0 is a valid RSI value (e.g., if all price changes are losses).
        if current_rsi == 0.0 and len(prices) < MIN_CANDLES_RSI:
             continue 

        # --- Signal Generation with Confirmation ---
        if price_deviation > FORECAST_DEVIATION_THRESHOLD and current_rsi < RSI_OVERBOUGHT:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif price_deviation < -FORECAST_DEVIATION_THRESHOLD and current_rsi > RSI_OVERSOLD:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals