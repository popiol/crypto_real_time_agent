from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from src.agent.models import BuySignal, SellSignal, MarketData, WarmCandle
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Suppress specific warnings from statsmodels, e.g., concerning convergence
warnings.filterwarnings("ignore", module="statsmodels.tsa.statespace.sarimax")
warnings.filterwarnings("ignore", module="statsmodels.base.model")

# --- Rule Parameters ---
# SARIMA Model Parameters (p,d,q)(P,D,Q,S)
# NOTE ON WARM_CANDLES LIMITATION:
# The `WarmCandle` data model provides at most 24 hourly candles.
# A seasonal period 'S' of 24 (for daily seasonality in hourly data) is not feasible
# with only 24 data points, as it would require at least two full seasons to observe patterns.
# Therefore, we use a non-seasonal ARIMA model by setting seasonal_order=(0,0,0,0).
# The rule's intent for "seasonal patterns" cannot be fully realized with the current data model.
SARIMA_P = 1
SARIMA_D = 1
SARIMA_Q = 1
SARIMA_SEASONAL_P = 0
SARIMA_SEASONAL_D = 0
SARIMA_SEASONAL_Q = 0
SARIMA_SEASONAL_S = 0 # Set to 0 to effectively use a non-seasonal ARIMA model

SARIMA_N_LOOKBACK = 20 # Number of past closing prices to fit SARIMA model. Max 23 with 24 candles.

# Adaptive Volatility Threshold Parameters
ATR_M_LOOKBACK = 14 # Lookback window for Average True Range (ATR)
VOLATILITY_K = 1.5  # Multiplier for ATR to set the volatility threshold

# Volume Confirmation Parameters
# CRITICAL NOTE ON VOLUME DATA LIMITATION:
# The `WarmCandle` data model, which provides OHLC data for candles, does NOT include volume per candle.
# The `Tick` data model includes 'volume_24h', which is a rolling 24-hour volume for the base currency,
# not specific candle volume, and thus unsuitable for calculating 'current_volume' and 'avg_volume'
# as required by the pseudocode for 'volume_spike_condition'.
# Therefore, the 'Volume Confirmation' step of the pseudocode CANNOT be implemented with the
# provided data models. The rule will proceed WITHOUT volume confirmation.
# This is a significant deviation from the pseudocode due to data limitations.
# VOLUME_L_LOOKBACK = 10 # Lookback window for average volume
# VOLUME_MULTIPLIER = 2.0 # Multiplier for average volume to detect a spike

# Minimum required candles for both SARIMA and ATR calculations
MIN_CANDLES = max(SARIMA_N_LOOKBACK, ATR_M_LOOKBACK) + 1 # +1 for current candle/price

RULE_ID = "3bb27e81-8d54-46fa-a726-2cc000d6a915"

def calculate_atr(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int) -> float:
    """
    Calculates the Average True Range (ATR) for a given period.
    Uses a simple moving average of True Ranges.
    """
    if len(high_prices) < period + 1:
        return np.nan # Not enough data to calculate ATR for the given period

    true_ranges = []
    for i in range(1, len(close_prices)):
        high = high_prices.iloc[i]
        low = low_prices.iloc[i]
        prev_close = close_prices.iloc[i-1]
        
        # True Range is the greatest of:
        # 1. Current High less Current Low
        # 2. Absolute value of Current High less Previous Close
        # 3. Absolute value of Current Low less Previous Close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    
    # Calculate simple moving average of the last 'period' True Ranges
    if len(true_ranges) < period:
        return np.nan # Still not enough true ranges for the period
    
    return pd.Series(true_ranges).iloc[-period:].mean()

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough warm candle data is available for calculations
        if len(warm_candles) < MIN_CANDLES:
            continue

        # Extract OHLC data into pandas Series for easier slicing and calculations
        close_prices = pd.Series([c.close for c in warm_candles])
        high_prices = pd.Series([c.high for c in warm_candles])
        low_prices = pd.Series([c.low for c in warm_candles])
        
        current_price = close_prices.iloc[-1]
        
        # --- 2. SARIMA Model Fitting & Forecasting ---
        # Use the last N closing prices for SARIMA fitting
        sarima_data = close_prices.iloc[-SARIMA_N_LOOKBACK:].dropna()
        
        # Ensure we have enough data points for SARIMA after dropping any NaNs
        if len(sarima_data) < SARIMA_N_LOOKBACK:
            continue

        forecast_price = np.nan
        try:
            # Fit SARIMA model (non-seasonal as per data limitations)
            model = SARIMAX(sarima_data,
                            order=(SARIMA_P, SARIMA_D, SARIMA_Q),
                            seasonal_order=(SARIMA_SEASONAL_P, SARIMA_SEASONAL_D, SARIMA_SEASONAL_Q, SARIMA_SEASONAL_S),
                            enforce_stationarity=False, # Relax constraints for potentially unstable series
                            enforce_invertibility=False)
            model_fit = model.fit(disp=False) # Suppress convergence output
            
            # Forecast the price for the next 1 period
            forecast_result = model_fit.forecast(steps=1)
            forecast_price = forecast_result.iloc[0]
            
        except (ValueError, np.linalg.LinAlgError) as e:
            # Handle cases where SARIMA fails to fit (e.g., singular matrix, non-convergence)
            # print(f"SARIMA fitting failed for {pair}: {e}") # Uncomment for debugging
            continue # Skip this pair if SARIMA fails or forecast is invalid
        except Exception as e: # Catch any other unexpected errors during SARIMA
            # print(f"Unexpected error during SARIMA for {pair}: {e}") # Uncomment for debugging
            continue

        if np.isnan(forecast_price):
            continue

        # --- 3. Adaptive Volatility Threshold ---
        atr = calculate_atr(high_prices, low_prices, close_prices, ATR_M_LOOKBACK)
        if np.isnan(atr) or atr <= 0: # ATR must be a positive value to be meaningful
            continue
        
        volatility_threshold = VOLATILITY_K * atr

        # --- 4. Volume Confirmation (SKIPPED due to data model limitations) ---
        # As noted at the top, the provided `WarmCandle` data model does not include volume.
        # Therefore, the volume confirmation step cannot be implemented as described in the pseudocode.
        # The signal generation will proceed based solely on SARIMA forecast and volatility.
        # volume_spike_condition = False
        # (Implementation for volume calculation if data were available)

        # --- 5. Signal Generation ---
        # current_price is already extracted from the latest candle
        
        # Emit Buy Signal if forecast is significantly higher than current price
        # (Volume confirmation condition removed due to data limitations)
        if forecast_price > current_price + volatility_threshold:
            signals.append(BuySignal(
                pair=pair,
                timestamp=warm_candles[-1].hour, # Use the timestamp of the last candle
                price=current_price,
                rule_id=RULE_ID,
                confidence=None # Confidence not specified in pseudocode
            ))
        # Else Emit Sell Signal if forecast is significantly lower than current price
        # (Volume confirmation condition removed due to data limitations)
        elif forecast_price < current_price - volatility_threshold:
            signals.append(SellSignal(
                pair=pair,
                timestamp=warm_candles[-1].hour,
                price=current_price,
                rule_id=RULE_ID,
                confidence=None # Confidence not specified in pseudocode
            ))
        # Else: No Signal.

    return signals