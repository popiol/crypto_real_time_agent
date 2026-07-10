from __future__ import annotations
import numpy as np
import pandas as pd
import statistics
from datetime import datetime
import warnings

# Suppress specific warnings from statsmodels
warnings.filterwarnings("ignore", module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning, message="Maximum Likelihood optimization failed to converge.")
warnings.filterwarnings("ignore", category=UserWarning, message="Non-stationary starting values detected.")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")


from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule Parameters ---
RULE_ID = "12a0cc1b-a372-4157-9131-55ae4933ea63"

# SARIMA Model Parameters (p, d, q) and (P, D, Q, s)
# Example: (1, 1, 1) and (1, 1, 0, 24) for daily seasonality on hourly data
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 0, 24) # s=24 for daily seasonality on hourly candles
FORECAST_HORIZON = 1 # Number of future periods to forecast (e.g., next hour)

# Volatility Calculation Parameters
VOLATILITY_WINDOW = 20 # Window for calculating historical volatility (in # of candles)
DEVIATION_MULTIPLIER = 1.5 # Multiplier for volatility threshold

# Minimum data points required for SARIMA fitting and volatility calculation
# At least 2 * s + p + P for SARIMA to be robust. For s=24, this means > 48.
# Also need enough for volatility window. Let's set a practical minimum.
MIN_CANDLES_FOR_SARIMA = 72 # e.g., 3 full days of hourly data for s=24
MIN_CANDLES_FOR_VOLATILITY = VOLATILITY_WINDOW + 1 # Need at least window+1 prices for std dev


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Ensure we have enough warm candle data for both SARIMA and volatility
        if len(warm_candles) < max(MIN_CANDLES_FOR_SARIMA, MIN_CANDLES_FOR_VOLATILITY):
            continue

        # Data Preparation for SARIMA
        # Sort candles by hour to ensure correct time series order
        warm_candles_sorted = sorted(warm_candles, key=lambda c: c.hour)
        
        # Extract closing prices and create a pandas Series with datetime index
        # This is crucial for statsmodels SARIMA
        prices = pd.Series(
            [c.close for c in warm_candles_sorted],
            index=[c.hour for c in warm_candles_sorted]
        )

        # Check if there's enough variation in prices for SARIMA
        # If all prices are the same, differencing will result in NaNs or zeros,
        # and SARIMA will likely fail.
        if prices.nunique() < 2:
            continue

        # Model Fitting and Forecasting
        forecast_price = None
        try:
            # Fit SARIMA model
            # Use suppress_warnings=True to prevent excessive output in some environments
            model = SARIMAX(
                prices,
                order=SARIMA_ORDER,
                seasonal_order=SARIMA_SEASONAL_ORDER,
                enforce_stationarity=False, # Allow differencing to handle non-stationarity
                enforce_invertibility=False # Allow MA parameters to be outside unit circle
            )
            # Use 'mle' for maximum likelihood estimation
            # Use a smaller maxiter for potentially faster, though less precise, convergence
            # This can help avoid convergence warnings/errors in highly volatile data
            results = model.fit(disp=False, maxiter=50) # disp=False suppresses convergence output

            # Forecast the next FORECAST_HORIZON step(s)
            forecast_result = results.forecast(steps=FORECAST_HORIZON)
            if not forecast_result.empty:
                forecast_price = forecast_result.iloc[-1] # Get the last forecast price
            else:
                continue # Forecast failed or returned empty

        except (ValueError, np.linalg.LinAlgError, RuntimeWarning) as e:
            # Handle cases where SARIMA model fitting fails (e.g., non-invertible, singular matrix)
            continue
        except Exception:
            # Catch other unexpected errors during SARIMA process
            continue

        if forecast_price is None:
            continue

        # Calculate Adaptive Deviation Threshold
        # Use the most recent prices for volatility calculation
        recent_prices_for_vol = np.array([c.close for c in warm_candles_sorted[-VOLATILITY_WINDOW:]])
        
        # Ensure enough data for std dev calculation
        if len(recent_prices_for_vol) < 2:
            continue

        # Calculate historical volatility (standard deviation of recent closing prices)
        historical_volatility = np.std(recent_prices_for_vol)

        # Handle cases where volatility is zero (e.g., all prices are the same)
        if historical_volatility == 0:
            continue
            
        adaptive_threshold = historical_volatility * DEVIATION_MULTIPLIER

        # Current price from the most recent tick
        if not hot_ticks:
            continue # No recent tick data
        current_price = hot_ticks[-1].last_price
        current_timestamp = hot_ticks[-1].polled_at

        # Signal Generation
        if forecast_price > current_price + adaptive_threshold:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=RULE_ID
            ))
        elif forecast_price < current_price - adaptive_threshold:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=RULE_ID
            ))

    return signals