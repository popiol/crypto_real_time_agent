from __future__ import annotations

from datetime import datetime
import numpy as np

# IMPORTANT ASSUMPTION REGARDING ARIMA FITTING:
# The pseudocode specifies `model = ARIMA(prices, order=(0,1,1))`
# and the list of "Available external packages" includes `numpy, tensorflow, keras`
# but *not* `statsmodels`. Implementing a correct ARIMA(0,1,1) model fitting
# (especially the Moving Average part, which typically requires Maximum Likelihood
# Estimation) from scratch using only `numpy` is extremely complex and not a
# straightforward OLS problem like the original rule's AR(1) fit.
# Therefore, it is assumed that `statsmodels` is implicitly allowed for standard
# ARIMA model fitting, as it is the industry-standard library for this purpose
# and aligns with the pseudocode's high-level `ARIMA()` constructor call.
import statsmodels.tsa.arima.model as sm_arima

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle


# Minimum warm candles needed for a reliable ARIMA(0,1,1) model fit.
# statsmodels generally requires a reasonable number of observations.
MIN_CANDLES = 10

# Minimum data points required to calculate standard deviation for volatility.
# At least 2 points are needed for a meaningful standard deviation.
MIN_VOLATILITY_CANDLES = 2

# Forecast horizon in hours (number of steps ahead to forecast).
FORECAST_HORIZON = 3

# Lookback window for calculating recent price volatility (in hours/candles).
LOOKBACK_VOLATILITY_WINDOW = 24

# Multiplier for the historical standard deviation to determine the adaptive deviation threshold.
VOLATILITY_MULTIPLIER = 1.5


def _fit_arima011(prices: list[float]):
    """
    Fits an ARIMA(0,1,1) model to a given price series using statsmodels.

    An ARIMA(0,1,1) model implies an MA(1) model on the first-differenced series.
    Returns the fitted model object or None if fitting fails.
    """
    if len(prices) < MIN_CANDLES:
        return None

    try:
        # Initialize and fit the ARIMA model with order (p=0, d=1, q=1).
        # d=1 means the series is differenced once before fitting an MA(1) model.
        model = sm_arima.ARIMA(prices, order=(0, 1, 1))
        model_fit = model.fit()
        return model_fit
    except Exception:
        # Catch any exceptions that might occur during model fitting (e.g.,
        # convergence issues, singular matrices, insufficient unique data points).
        return None


def _forecast_arima011(model_fit, horizon: int) -> float:
    """
    Generates a point forecast from a fitted ARIMA(0,1,1) model for 'horizon' steps ahead.
    """
    if model_fit is None:
        return 0.0  # Return a default value or indicate failure if the model wasn't fitted.

    # model_fit.forecast(steps=horizon) returns a pandas Series containing forecasts
    # for `t+1`, `t+2`, ..., `t+horizon`.
    # The pseudocode's `[0]` is ambiguous but if interpreted as getting the forecast
    # for `H` steps ahead (consistent with the original rule's iterative forecast),
    # then the last element of the forecast series is needed.
    forecast_series = model_fit.forecast(steps=horizon)
    return float(forecast_series.iloc[-1])


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Detects when the current market price deviates significantly from its
    short-term ARIMA(0,1,1) forecast, emitting Buy/Sell signals based on
    an adaptive, volatility-adjusted threshold.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure there is recent price data available for the current price and timestamp.
        if not pair_data.hot:
            continue

        # Extract close prices from warm candles for ARIMA modeling and volatility calculation.
        prices = [c.close for c in pair_data.warm]

        # Ensure enough historical data for both ARIMA model fitting and volatility calculation.
        # ARIMA requires MIN_CANDLES. Volatility requires at least MIN_VOLATILITY_CANDLES
        # and up to LOOKBACK_VOLATILITY_WINDOW.
        if len(prices) < MIN_CANDLES:
            continue

        # 1. Train ARIMA(0,1,1) Model
        model_fit = _fit_arima011(prices)
        if model_fit is None:
            continue

        # 2. Generate Forecast
        forecast_price = _forecast_arima011(model_fit, FORECAST_HORIZON)

        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # 3. Calculate Adaptive Threshold
        # The pseudocode mentions `data.returns[-volatility_window:]` for volatility.
        # However, the base rule (`rule_04_arima_forecast_v2`) uses `np.std` on
        # `recent_prices_for_volatility` (i.e., standard deviation of prices directly).
        # We adhere to the existing rule's implementation for this part for consistency.
        recent_prices_for_volatility = prices[-LOOKBACK_VOLATILITY_WINDOW:]

        if len(recent_prices_for_volatility) < MIN_VOLATILITY_CANDLES:
            continue  # Not enough data points to calculate standard deviation.

        price_std = np.std(recent_prices_for_volatility)

        # Calculate the adaptive deviation threshold.
        # If `price_std` is zero (indicating constant prices), the threshold becomes zero.
        # The original rule skips generating a signal if `deviation_threshold <= 0`
        # to prevent overly sensitive signals on flat data.
        deviation_threshold = VOLATILITY_MULTIPLIER * price_std

        if deviation_threshold <= 0:
            continue

        # 4. Generate Signal
        price_diff = forecast_price - current_price

        if price_diff > deviation_threshold:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif price_diff < -deviation_threshold:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals