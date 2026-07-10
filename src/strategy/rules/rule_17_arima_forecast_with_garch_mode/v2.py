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

# Confidence interval alpha level (e.g., 0.05 for 95% CI)
ALPHA = 0.05
# Z-score for a two-tailed alpha level (e.g., for alpha=0.05, Z-score is approx 1.96)
# This can be calculated using scipy.stats.norm.ppf(1 - ALPHA / 2) but for simplicity
# and to avoid additional dependencies, it's hardcoded here.
Z_SCORE_FOR_CI = 1.96 

# Unique identifier for this rule
RULE_ID = "3e54031e-bbd8-4986-ba7e-d15f62510866"


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the ARIMA-GARCH Forecast with Model Confidence Interval Thresholds rule.

    This rule refines the previous ARIMA-GARCH approach by directly using the model's
    predicted confidence intervals to generate buy and sell signals. A Buy signal is
    emitted if the current price falls below the lower bound of the forecasted
    confidence interval, indicating significant undervaluation. A Sell signal is
    emitted if the current price rises above the upper bound, suggesting significant
    overvaluation. The confidence interval is constructed using the ARIMA forecast mean
    and the GARCH-predicted volatility as the standard deviation of the forecast error.
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
            arima_model = ARIMA(arima_prices, order=ARIMA_ORDER)
            arima_fit = arima_model.fit()
            # Forecast the next period's price mean
            p_forecast = arima_fit.forecast(steps=1)[0]
        except Exception:
            # If ARIMA model fails to fit (e.g., convergence issues), skip this pair
            continue
        
        if p_forecast is None or not np.isfinite(p_forecast) or p_forecast <= 0:
            continue

        # Step 2: Estimate future volatility using GARCH(1,1) model
        sigma_forecast_log_returns: float | None = None
        try:
            garch_model = arch_model(garch_returns, vol='Garch', p=GARCH_ORDER[0], q=GARCH_ORDER[1], dist='normal')
            garch_fit = garch_model.fit(disp='off')
            
            # Forecast the next period's conditional variance, then take sqrt for standard deviation
            forecasts = garch_fit.forecast(horizon=1)
            sigma_forecast_log_returns = np.sqrt(forecasts.variance.values[-1, 0])

        except Exception:
            # If GARCH model fails to fit (e.g., convergence issues), skip this pair
            continue

        if (sigma_forecast_log_returns is None or 
                not np.isfinite(sigma_forecast_log_returns) or 
                sigma_forecast_log_returns <= 0): # Standard deviation must be positive
            continue

        # Step 3: Construct confidence interval using ARIMA forecast mean and GARCH volatility
        # We approximate the standard deviation of the price forecast error by scaling
        # the GARCH-predicted log return volatility by the forecasted price.
        sigma_price_error = sigma_forecast_log_returns * p_forecast

        # Calculate the lower and upper bounds of the confidence interval
        lower_ci = p_forecast - Z_SCORE_FOR_CI * sigma_price_error
        upper_ci = p_forecast + Z_SCORE_FOR_CI * sigma_price_error

        # Step 4: Generate signals based on confidence interval thresholds
        if current_price < lower_ci:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID,
            ))
        elif current_price > upper_ci:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID,
            ))

    return signals