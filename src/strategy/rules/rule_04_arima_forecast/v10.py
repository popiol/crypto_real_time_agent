from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel, Field

# Assuming these models are available in a src.agent.models path or similar.
# For a self-contained module, I'll define them here as per the prompt's "Available data models".

class Tick(BaseModel):
    """A single poll snapshot for one currency pair."""

    pair: str
    polled_at: datetime

    # Last trade
    last_price: float

    # Best bid / ask from Ticker
    bid_price: float
    bid_volume: float
    ask_price: float
    ask_volume: float

    # 24-hour rolling volume in base currency (from Kraken Ticker v[1])
    volume_24h: float = 0.0

    # Derived
    mid_price: float
    spread_abs: float  # ask - bid
    spread_rel: float  # (ask - bid) / mid  * 100  (%)

    # Top-5 order book (from Depth endpoint)
    # OrderBook definition is missing in prompt, but not used in this rule, so can be omitted.
    # For completeness, if it were used, it would need to be defined.
    # For now, setting it to Any or just leaving it as None.
    order_book: dict | None = None


class WarmCandle(BaseModel):
    hour: datetime
    open_price: float
    high: float
    low: float
    close: float
    avg_spread_rel: float = 0.0


class ColdMonth(BaseModel):
    month: str  # "YYYY-MM"
    min_price: float
    max_price: float
    avg_price: float
    avg_daily_spread: float
    candle_count: int
    last_candle_hour: datetime


class PairData(BaseModel):
    hot: list[Tick] = Field(
        default=[],
        description="TTL-capped; ~300 ticks at 1 poll/sec with default 300s retention",
    )
    warm: list[WarmCandle] = Field(
        default=[], description="At most 24 entries (last 24 hourly candles)"
    )
    cold: list[ColdMonth] = Field(
        default=[], description="One entry per calendar month; unbounded"
    )


class BuySignal(BaseModel):
    pair: str
    timestamp: datetime
    price: float
    rule_id: str = ""
    confidence: float | None = None


class SellSignal(BaseModel):
    pair: str
    timestamp: datetime
    price: float
    rule_id: str = ""
    confidence: float | None = None


MarketData = dict[str, PairData]


# Minimum warm candles needed for a reliable fit for current timeframe ARIMA
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# Lookback window for calculating recent volatility (in hours/candles)
LOOKBACK_VOLATILITY_WINDOW = 24

# Multiplier for the historical standard deviation to determine the deviation threshold
VOLATILITY_MULTIPLIER = 1.5

# Multi-Timeframe RSI Confirmation Parameters
HIGHER_TIMEFRAME_FACTOR = 4  # e.g., 4x current timeframe (4-hour candles if current is 1-hour)

# RSI period for the higher timeframe.
# Note: The `warm` data (max 24 hourly candles) limits the number of aggregated
# 4-hour candles to 6 (24 / 4). A standard RSI(14) would require at least 15 candles.
# We adjust this value to `5` to allow calculation with the available data.
# If more historical `warm` data were available, this could be 14 or higher.
RSI_PERIOD_HIGHER_TF = 5

# --- MODIFICATION START ---
# Relaxed RSI thresholds as per the rule idea
RSI_OVERSOLD_THRESHOLD = 40
RSI_OVERBOUGHT_THRESHOLD = 60
# --- MODIFICATION END ---


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


def _aggregate_closes_by_factor(candles: list[WarmCandle], factor: int) -> list[float]:
    """Aggregates close prices of WarmCandle (hourly) into higher timeframe blocks.
    Takes the close price of the last candle in each complete block of `factor` candles.
    """
    if not candles or factor <= 0:
        return []

    aggregated_closes = []
    # Iterate over full blocks of `factor` candles
    for i in range(len(candles) // factor):
        # The close of the aggregated candle is the close of the last candle in the block
        aggregated_closes.append(candles[(i + 1) * factor - 1].close)
    
    return aggregated_closes


def _calculate_rsi(prices: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI) for a given price series."""
    if len(prices) < period + 1:
        return None

    # Calculate initial gains and losses
    gains = [0.0] * (len(prices) - 1)
    losses = [0.0] * (len(prices) - 1)

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains[i - 1] = change
        else:
            losses[i - 1] = abs(change)

    # Calculate initial average gain and loss over the first 'period' changes
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Calculate initial RS and RSI
    if avg_loss == 0:
        rs = float('inf')  # All gains, no losses -> RSI 100
    else:
        rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # Apply Wilder's smoothing for subsequent periods to get the latest RSI
    for i in range(period, len(gains)): # Iterate through the remaining changes
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

        if avg_loss == 0:
            rs = float('inf')
        else:
            rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    return rsi


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Current timeframe data (hourly candles)
        current_tf_prices = [c.close for c in pair_data.warm]
        
        # Higher timeframe data (e.g., 4-hour candles from hourly data)
        higher_tf_prices = _aggregate_closes_by_factor(pair_data.warm, HIGHER_TIMEFRAME_FACTOR)

        # 2. If insufficient data, return "NO_SIGNAL"
        # Check for sufficient data for ARIMA forecast and volatility calculation
        if (
            len(current_tf_prices) < MIN_CANDLES
            or len(current_tf_prices) < LOOKBACK_VOLATILITY_WINDOW
            or not pair_data.hot # Need hot data for current price and timestamp
        ):
            continue

        # Check for sufficient data for higher timeframe RSI calculation
        # Need at least period + 1 data points for RSI
        if len(higher_tf_prices) < RSI_PERIOD_HIGHER_TF + 1:
            continue

        # 3. Calculate ARIMA(arima_order) forecast for the next period
        phi, intercept, sigma = _fit_arima110(current_tf_prices)

        # Skip if ARIMA model is degenerate
        if sigma == 0 or not math.isfinite(phi) or not math.isfinite(intercept):
            continue

        forecast_price = _forecast(current_tf_prices, phi, intercept, FORECAST_HORIZON)
        
        # 4. Determine current_price
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # 5. Calculate adaptive_threshold
        recent_prices_for_volatility = current_tf_prices[-LOOKBACK_VOLATILITY_WINDOW:]
        price_std = np.std(recent_prices_for_volatility)
        
        deviation_threshold = VOLATILITY_MULTIPLIER * price_std
        
        # Avoid signals based on zero or near-zero volatility
        if deviation_threshold <= 0:
            continue

        # 6. Calculate higher_timeframe_rsi
        higher_timeframe_rsi = _calculate_rsi(higher_tf_prices, RSI_PERIOD_HIGHER_TF)
        if higher_timeframe_rsi is None:
            # This case should ideally be covered by the earlier length check for higher_tf_prices,
            # but it's a safe guard.
            continue 

        # 7-10. Define signal conditions based on ARIMA forecast and RSI confirmation
        buy_signal_condition_arima = forecast_price > (current_price + deviation_threshold)
        sell_signal_condition_arima = forecast_price < (current_price - deviation_threshold)

        # --- MODIFICATION START ---
        # Updated RSI confirmation thresholds
        buy_signal_condition_rsi = higher_timeframe_rsi < RSI_OVERSOLD_THRESHOLD  # Now < 40
        sell_signal_condition_rsi = higher_timeframe_rsi > RSI_OVERBOUGHT_THRESHOLD # Now > 60
        # --- MODIFICATION END ---

        # 11-12. Combine conditions for final signal
        if buy_signal_condition_arima and buy_signal_condition_rsi:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif sell_signal_condition_arima and sell_signal_condition_rsi:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals