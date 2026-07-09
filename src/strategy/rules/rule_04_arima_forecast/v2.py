from __future__ import annotations

import math
import statistics

from src.agent.models import BuySignal, MarketData, PairData, SellSignal

# --- Constants for ARIMA ---
# Minimum warm candles needed for a reliable ARIMA fit.
# Needs at least 3 candles for ARIMA(1,1,0) (2 diffs for AR(1) regression).
MIN_CANDLES_ARIMA = 10

# Forecast horizon in hours (pseudocode specifies 1 for this rule)
FORECAST_HORIZON = 1

# Signal fires when forecast exceeds current price by > this many forecast std devs (pseudocode specifies 2.0)
SIGNAL_THRESHOLD = 2.0

# --- Constants for RSI ---
RSI_PERIOD = 14
OVERSOLD_THRESHOLD = 30
OVERBOUGHT_THRESHOLD = 70

# Minimum warm candles needed for RSI calculation (RSI_PERIOD + 1)
MIN_CANDLES_RSI = RSI_PERIOD + 1

# Overall minimum candles required for both ARIMA and RSI
MIN_CANDLES = max(MIN_CANDLES_ARIMA, MIN_CANDLES_RSI)


def _fit_arima110(prices: list[float]) -> tuple[float, float, float]:
    """Fit ARIMA(1,1,0) to a price series via OLS on first differences.

    Returns (phi, intercept, sigma_residual) where phi is the AR(1) coefficient
    on the differenced series and sigma_residual is in price units.
    """
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    x = diffs[:-1]
    y = diffs[1:]
    n = len(x)

    if n < 2:  # Need at least 2 data points for regression (x, y)
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
    # statistics.stdev requires at least 2 data points. If only 1 residual, use its absolute value.
    sigma = statistics.stdev(residuals) if n >= 2 else abs(residuals[0]) if residuals else 0.0

    return phi, intercept, sigma


def _forecast(prices: list[float], phi: float, intercept: float, horizon: int) -> float:
    """Iterate the ARIMA(1,1,0) recursion h steps ahead."""
    if len(prices) < 2:
        # Not enough data to compute the initial difference
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


def _calculate_rsi(prices: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI).

    Returns the latest RSI value or None if not enough data.
    """
    if len(prices) <= period:
        return None

    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    # Calculate initial average gain and loss over the first 'period' changes
    initial_gains = [c for c in changes[0:period] if c > 0]
    initial_losses = [abs(c) for c in changes[0:period] if c < 0]

    avg_gain = sum(initial_gains) / period
    avg_loss = sum(initial_losses) / period

    # Apply smoothing for subsequent changes
    for i in range(period, len(changes)):
        current_change = changes[i]
        gain = current_change if current_change > 0 else 0
        loss = abs(current_change) if current_change < 0 else 0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    
    # Handle cases where avg_loss is zero
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0 # If no losses, RSI is 100. If no gains and no losses, it's 50.
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        if len(pair_data.warm) < MIN_CANDLES or not pair_data.hot:
            continue

        prices = [c.close for c in pair_data.warm]
        
        # --- ARIMA Calculation ---
        phi, intercept, sigma = _fit_arima110(prices)

        if sigma <= 0 or not math.isfinite(phi):
            # If sigma is non-positive, there's no price volatility or residuals.
            # If phi is not finite, the model is degenerate.
            continue

        forecast_price = _forecast(prices, phi, intercept, FORECAST_HORIZON)
        current_price = pair_data.hot[-1].last_price # Use the most recent price from hot data
        std = _forecast_std(phi, sigma, FORECAST_HORIZON)
        ts = pair_data.hot[-1].polled_at

        if std <= 0: # Avoid division by zero or non-sensical std
            continue

        deviation = (forecast_price - current_price) / std

        # --- RSI Calculation ---
        rsi_value = _calculate_rsi(prices, RSI_PERIOD)
        if rsi_value is None:
            continue # Not enough data for RSI calculation

        # --- Signal Logic with Confirmation ---
        # Buy Signal: ARIMA forecast significantly higher AND RSI oversold
        if deviation > SIGNAL_THRESHOLD and rsi_value < OVERSOLD_THRESHOLD:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Sell Signal: ARIMA forecast significantly lower AND RSI overbought
        elif deviation < -SIGNAL_THRESHOLD and rsi_value > OVERBOUGHT_THRESHOLD:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals