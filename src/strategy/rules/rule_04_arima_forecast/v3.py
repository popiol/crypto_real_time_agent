from __future__ import annotations

import math
import statistics
from datetime import datetime

from pydantic import BaseModel, Field


# Data models (as provided in the prompt for self-containment)
class Tick(BaseModel):
    pair: str
    polled_at: datetime
    last_price: float
    bid_price: float
    bid_volume: float
    ask_price: float
    ask_volume: float
    volume_24h: float = 0.0
    mid_price: float
    spread_abs: float
    spread_rel: float
    order_book: dict | None = None  # Changed from OrderBook to dict as model not provided


class WarmCandle(BaseModel):
    hour: datetime
    open_price: float
    high: float
    low: float
    close: float
    avg_spread_rel: float = 0.0
    volume: float = Field(default=0.0, description="Average volume_24h of ticks within this hour (proxy for relative market activity)")


class ColdMonth(BaseModel):
    month: str
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


# Minimum warm candles needed for a reliable fit
MIN_CANDLES = 10

# Forecast horizon in hours
FORECAST_HORIZON = 3

# ARIMA model parameters
ARIMA_ORDER = (1, 1, 0)  # Not directly used in the _fit_arima110 but good for context
ARIMA_LOOKBACK = 60  # bars for ARIMA model training

# Volatility window parameters
VOLATILITY_WINDOW_SHORT = 20  # bars for recent volatility
VOLATILITY_WINDOW_LONG = 100  # bars for long-term average volatility

# Baseline deviation multiplier
BASE_K = 2.0

# New parameters for adaptive K-factor smoothing
N_PERIODS_K_AVG = 20  # Number of periods for the moving average of K-factor
VOLATILITY_SMOOTHING_FACTOR = 0.8  # Weight for current volatility


def _fit_arima110(prices: list[float]) -> tuple[float, float, float]:
    """Fit ARIMA(1,1,0) to a price series via OLS on first differences.

    Returns (phi, intercept, sigma_residual) where phi is the AR(1) coefficient
    on the differenced series and sigma_residual is in price units.
    """
    if len(prices) < 2:  # Need at least two prices to calculate a difference
        return 0.0, 0.0, 0.0

    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    if len(diffs) < 2:  # Need at least two differences for OLS (x,y pairs)
        # If only one diff, we can't fit AR(1). Assume no AR component,
        # and forecast is just the mean of the diff (or 0 if no diffs).
        if len(diffs) == 1:
            return 0.0, diffs[0], 0.0
        return 0.0, 0.0, 0.0

    x = diffs[:-1]
    y = diffs[1:]
    n = len(x)

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
        return prices[-1] if prices else 0.0  # Cannot forecast without prior differences

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


def _safe_stdev(data: list[float]) -> float:
    """Calculates standard deviation, returning 0.0 if data is insufficient or constant."""
    if len(data) < 2:
        return 0.0
    try:
        return statistics.stdev(data)
    except statistics.StatisticsError:
        # This typically happens if all data points are identical.
        return 0.0


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        if not pair_data.hot:
            continue

        prices_warm = [c.close for c in pair_data.warm]
        num_warm_candles = len(prices_warm)

        # Ensure enough data for overall processing (e.g., for ARIMA and initial volatility checks)
        if num_warm_candles < MIN_CANDLES:
            continue

        # --- Calculate the adaptive K-factor using volatility-weighted average ---

        # Lists to store historical K-factors and their corresponding volatilities for weighting
        historical_k_factors_for_avg: list[float] = []
        historical_volatilities_for_weighting: list[float] = []

        # We need enough data to calculate both short and long term volatility for each historical point.
        # This is the minimum length of `sub_prices` required to calculate a reliable K-factor.
        min_required_for_full_k_calc = max(VOLATILITY_WINDOW_SHORT, VOLATILITY_WINDOW_LONG)

        # Iterate over the warm candles to build the history for the weighted average.
        # We need to consider up to N_PERIODS_K_AVG historical points,
        # starting from the earliest point for which we can calculate a K-factor.
        # The loop starts from an index that ensures enough preceding data for volatility calculations.
        start_index = max(0, num_warm_candles - N_PERIODS_K_AVG, min_required_for_full_k_calc - 1)

        for i in range(start_index, num_warm_candles):
            # `sub_prices` represents all prices up to and including the current historical candle `i`.
            # This allows calculating rolling volatility windows.
            sub_prices = prices_warm[:i+1]
            sub_num_candles = len(sub_prices)

            # Recalculate actual window sizes for this historical point
            hist_vol_short_window_actual = min(VOLATILITY_WINDOW_SHORT, sub_num_candles)
            hist_vol_long_window_actual = min(VOLATILITY_WINDOW_LONG, sub_num_candles)

            # Need at least 2 points for stdev, and enough for the respective windows
            if hist_vol_short_window_actual < 2 or hist_vol_long_window_actual < 2:
                continue

            hist_recent_prices = sub_prices[-hist_vol_short_window_actual:]
            hist_long_term_prices = sub_prices[-hist_vol_long_window_actual:]

            hist_current_volatility = _safe_stdev(hist_recent_prices)
            hist_avg_volatility = _safe_stdev(hist_long_term_prices)

            # Calculate the instantaneous K-factor for this historical point
            hist_k_factor = BASE_K  # Default fallback for this historical point
            if hist_avg_volatility > 0:
                hist_k_factor = BASE_K * (1 + (hist_current_volatility - hist_avg_volatility) / hist_avg_volatility)
                # Apply minimum clamp as in the original rule
                if hist_k_factor < 0.1:
                    hist_k_factor = 0.1
            
            historical_k_factors_for_avg.append(hist_k_factor)
            historical_volatilities_for_weighting.append(hist_current_volatility)  # Using current volatility for weighting


        weighted_k_factor_sum = 0.0
        total_weight = 0.0
        adaptive_k_factor = BASE_K  # Default fallback for adaptive_k_factor

        num_historical_points = len(historical_k_factors_for_avg)

        if num_historical_points == 0:
            # If no historical K-factors could be calculated, fall back to the original rule's adaptive K-factor
            # for the current moment. This re-uses the logic for the most recent data.
            volatility_short_window_actual = min(VOLATILITY_WINDOW_SHORT, num_warm_candles)
            volatility_long_window_actual = min(VOLATILITY_WINDOW_LONG, num_warm_candles)

            if volatility_short_window_actual >= 2 and volatility_long_window_actual >= 2:
                recent_prices_volatility = prices_warm[-volatility_short_window_actual:]
                long_term_prices_volatility = prices_warm[-volatility_long_window_actual:]

                current_volatility = _safe_stdev(recent_prices_volatility)
                avg_volatility = _safe_stdev(long_term_prices_volatility)

                if avg_volatility > 0:
                    adaptive_k_factor = BASE_K * (1 + (current_volatility - avg_volatility) / avg_volatility)
                    if adaptive_k_factor < 0.1:
                        adaptive_k_factor = 0.1
        else:
            # Apply volatility-weighted moving average for the K-factor
            # Iterate through the collected historical points from most recent to oldest, up to N_PERIODS_K_AVG
            for k in range(min(N_PERIODS_K_AVG, num_historical_points)):
                # Get the k-th most recent historical point
                idx = num_historical_points - 1 - k
                
                hist_k_factor = historical_k_factors_for_avg[idx]
                hist_volatility = historical_volatilities_for_weighting[idx]

                # Calculate weight as per pseudocode: EXP(-HISTORICAL_VOLATILITY / VOLATILITY_SMOOTHING_FACTOR)
                # Handle VOLATILITY_SMOOTHING_FACTOR = 0 to prevent division by zero.
                if VOLATILITY_SMOOTHING_FACTOR <= 0:
                    # If smoothing factor is non-positive, treat all weights as uniform (1.0).
                    weight = 1.0
                else:
                    weight = math.exp(-hist_volatility / VOLATILITY_SMOOTHING_FACTOR)
                
                weighted_k_factor_sum += hist_k_factor * weight
                total_weight += weight

            if total_weight > 0:
                adaptive_k_factor = weighted_k_factor_sum / total_weight
                # Apply minimum clamp to the final adaptive K-factor
                if adaptive_k_factor < 0.1:
                    adaptive_k_factor = 0.1
            else:
                # Fallback if all weights summed to zero (e.g., due to extreme volatility or bad smoothing factor)
                # In this case, use the most recent instantaneous K-factor calculated, or BASE_K
                if num_historical_points > 0:
                    adaptive_k_factor = historical_k_factors_for_avg[-1]
                else:
                    adaptive_k_factor = BASE_K  # Should be covered by `num_historical_points == 0` block, but for robustness


        # --- Train and forecast with ARIMA model ---
        arima_lookback_actual = min(ARIMA_LOOKBACK, num_warm_candles)

        # ARIMA model requires at least 2 prices to calculate differences for fitting
        # and MIN_CANDLES for a reliable fit.
        if arima_lookback_actual < MIN_CANDLES:
            continue
        
        historical_prices_arima = prices_warm[-arima_lookback_actual:]

        phi, intercept, sigma = _fit_arima110(historical_prices_arima)

        if sigma <= 0 or not math.isfinite(phi):
            continue

        forecast_price = _forecast(historical_prices_arima, phi, intercept, FORECAST_HORIZON)
        current_price = pair_data.hot[-1].last_price
        forecast_std_dev = _forecast_std(phi, sigma, FORECAST_HORIZON)
        ts = pair_data.hot[-1].polled_at

        if forecast_std_dev <= 0:
            continue

        # --- Generate signal ---
        normalized_deviation = (forecast_price - current_price) / forecast_std_dev

        if normalized_deviation > adaptive_k_factor:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        elif normalized_deviation < -adaptive_k_factor:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals