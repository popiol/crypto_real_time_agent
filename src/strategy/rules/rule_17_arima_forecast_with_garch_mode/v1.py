# Assuming statsmodels and arch are available, as they are standard for ARIMA/GARCH.
# If these libraries are not permitted by the environment (i.e., only numpy, tensorflow, keras),
# this rule cannot be implemented as described without significantly more complex custom implementations
# of ARIMA and GARCH models using the allowed libraries.
from __future__ import annotations
from datetime import datetime
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

# Parameters
# ARIMA model order (p, d, q)
ARIMA_ORDER = (1, 1, 0)
# Number of hourly candles used for ARIMA price history
ARIMA_WINDOW_SIZE = 20 

# GARCH model order (p, q)
GARCH_ORDER = (1, 1)
# Number of hourly log returns used for GARCH volatility estimation.
# This means GARCH_WINDOW_SIZE + 1 prices are needed to calculate GARCH_WINDOW_SIZE returns.
GARCH_WINDOW_SIZE = 20 

# Multiplier for GARCH-predicted standard deviation to set the deviation threshold
DEVIATION_MULTIPLIER = 1.5 

# Unique identifier for this rule
RULE_ID = "b8de5385-bad7-45ca-a45c-57463deff5b1"


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the ARIMA Forecast with GARCH-modeled Adaptive Volatility Threshold rule.

    This rule detects market conditions where an ARIMA(1,1,0) model's price forecast
    significantly deviates from the current market price, where the significance
    threshold is dynamically determined by a GARCH(1,1) model's estimate of
    future price volatility. It emits a Buy signal if the ARIMA forecast is higher
    than the current price plus the GARCH-predicted volatility threshold, and a
    Sell signal if it's lower than the current price minus the GARCH-predicted
    volatility threshold.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Get current price and timestamp from the latest tick
        ticks = pair_data.hot
        if not ticks:
            continue
        current_price = ticks[-1].last_price
        timestamp = ticks[-1].polled_at

        # Get historical hourly candles for ARIMA and GARCH
        warm_candles = pair_data.warm

        # Determine the minimum number of candles required for both models
        # ARIMA needs ARIMA_WINDOW_SIZE prices.
        # GARCH needs GARCH_WINDOW_SIZE returns, which means GARCH_WINDOW_SIZE + 1 prices.
        required_candles_for_arima = ARIMA_WINDOW_SIZE
        required_candles_for_garch_prices = GARCH_WINDOW_SIZE + 1
        
        required_total_candles = max(required_candles_for_arima, required_candles_for_garch_prices)

        if len(warm_candles) < required_total_candles:
            # Not enough historical data to fit models
            continue

        # Extract closing prices for ARIMA model
        # Use the most recent ARIMA_WINDOW_SIZE candles
        arima_prices = np.array([c.close for c in warm_candles[-ARIMA_WINDOW_SIZE:]], dtype=np.float64)

        # Extract closing prices for GARCH log returns calculation
        # Use the most recent (GARCH_WINDOW_SIZE + 1) candles
        garch_raw_prices = np.array([c.close for c in warm_candles[-required_candles_for_garch_prices:]], dtype=np.float64)

        # Basic data validation to prevent errors in log/model fitting
        if not np.all(np.isfinite(arima_prices)) or np.any(arima_prices <= 0):
            continue
        if not np.all(np.isfinite(garch_raw_prices)) or np.any(garch_raw_prices <= 0):
            continue

        # Calculate log returns for GARCH model
        # garch_returns will have GARCH_WINDOW_SIZE elements
        garch_returns = np.diff(np.log(garch_raw_prices))
        if not np.all(np.isfinite(garch_returns)):
            continue
        
        # Ensure there's enough data for GARCH returns after differencing (e.g., at least 2 points for 1 return)
        # and enough for GARCH model order (p+q+1 observations needed for (p,q) model typically)
        if len(garch_returns) < GARCH_ORDER[0] + GARCH_ORDER[1] + 1:
             continue

        # Step 1: Forecast price using ARIMA(1,1,0) model
        p_forecast: float | None = None
        try:
            # ARIMA model expects enough data points for its order (p+d+q).
            # With ARIMA_WINDOW_SIZE = 20 and order (1,1,0), this should be sufficient.
            arima_model = ARIMA(arima_prices, order=ARIMA_ORDER)
            arima_fit = arima_model.fit()
            # Forecast the next period's price
            p_forecast = arima_fit.forecast(steps=1)[0]
        except Exception:
            # If ARIMA model fails to fit (e.g., convergence issues), skip this pair
            continue
        
        if p_forecast is None or not np.isfinite(p_forecast):
            continue

        # Step 2: Estimate future volatility using GARCH(1,1) model
        sigma_forecast_log_returns: float | None = None
        try:
            # arch_model expects returns data.
            # 'vol='Garch'' specifies GARCH model, p and q are orders.
            # 'dist='normal'' assumes normal distribution for innovations.
            garch_model = arch_model(garch_returns, vol='Garch', p=GARCH_ORDER[0], q=GARCH_ORDER[1], dist='normal')
            garch_fit = garch_model.fit(disp='off') # disp='off' to suppress solver output
            
            # Forecast the next period's conditional variance, then take sqrt for standard deviation
            forecasts = garch_fit.forecast(horizon=1)
            # The variance is in the last row (corresponding to the last observation used for fitting)
            # and first column (for horizon=1).
            sigma_forecast_log_returns = np.sqrt(forecasts.variance.values[-1, 0])
            
            # Convert standard deviation of log returns to a price deviation.
            # (sigma of log returns) * current_price approximates the standard deviation of price changes.
            volatility_threshold = sigma_forecast_log_returns * DEVIATION_MULTIPLIER * current_price

        except Exception:
            # If GARCH model fails to fit (e.g., convergence issues), skip this pair
            continue

        if (sigma_forecast_log_returns is None or 
                not np.isfinite(sigma_forecast_log_returns) or 
                sigma_forecast_log_returns <= 0): # Standard deviation must be positive
            continue

        # Step 3: Generate Signal
        if p_forecast > (current_price + volatility_threshold):
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID,
            ))
        elif p_forecast < (current_price - volatility_threshold):
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID,
            ))

    return signals