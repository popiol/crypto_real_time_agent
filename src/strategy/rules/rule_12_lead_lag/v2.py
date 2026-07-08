from __future__ import annotations

import math
import numpy as np
from datetime import datetime, timedelta
import logging

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick

# --- Constants ---
LOOKBACK_WINDOW_BARS = 200  # Number of warm candles for historical analysis
REBALANCE_FREQUENCY_BARS = 50  # How often to re-evaluate cointegration, hedge ratio, optimal lag
ENTRY_THRESHOLD_STD_DEV = 2.0  # Multiplier for standard deviation for entry signals
EXIT_THRESHOLD_STD_DEV = 0.5   # Multiplier for standard deviation for exiting positions (not fully implemented for simplicity in signal, but good for context)
# MIN_COINTEGRATION_P_VALUE = 0.05 # Cannot implement ADF test robustly without statsmodels. This constant is effectively ignored.
MAX_LAG_SEARCH_BARS = 10     # Maximum lead time in bars to consider for optimal lag
MIN_CANDLES_FOR_ANALYSIS = LOOKBACK_WINDOW_BARS + MAX_LAG_SEARCH_BARS # Minimum warm candles required for full lookback
MIN_CORRELATION_FOR_PAIR = 0.5 # Minimum Pearson r for lead-lag detection (retained from original rule)
MIN_SPREAD_STD_DEV = 1e-6 # Minimum standard deviation for spread to avoid division by zero or trivial cases

# --- Global Cache for Lead-Lag Parameters ---
# Stores: (hedge_ratio, optimal_lag, mean_spread, std_spread, correlation) for each (asset_A, asset_B) pair
_cached_lead_lag_params: dict[tuple[str, str], dict] = {}
_last_rebalance_warm_candle_hour: datetime | None = None
_last_warm_candle_count: int = 0

# --- Logger ---
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _get_prices(candles: list[WarmCandle]) -> np.ndarray:
    """Extract close prices from warm candles."""
    return np.array([c.close for c in candles], dtype=np.float64)

def _get_returns(closes: np.ndarray) -> np.ndarray:
    """Calculate percentage returns from close prices."""
    if len(closes) < 2:
        return np.array([])
    return (closes[1:] - closes[:-1]) / np.where(closes[:-1] != 0, closes[:-1], 1.0)

def _calculate_ols_hedge_ratio(prices_A: np.ndarray, prices_B: np.ndarray) -> float:
    """
    Performs OLS regression: prices_B = beta * prices_A + intercept
    Returns beta (hedge ratio).
    """
    if len(prices_A) < 2 or len(prices_B) < 2 or len(prices_A) != len(prices_B):
        return np.nan

    try:
        # Fit y = m*x + c, where y is prices_B and x is prices_A
        # The slope 'm' is the hedge ratio (beta)
        slope, _ = np.polyfit(prices_A, prices_B, 1)
        return slope
    except np.linalg.LinAlgError:
        logger.warning("OLS regression failed for prices. Returning NaN.")
        return np.nan
    except ValueError as e:
        logger.warning(f"OLS regression failed with ValueError: {e}. Returning NaN.")
        return np.nan

def _calculate_optimal_lag_returns(returns_A: np.ndarray, returns_B: np.ndarray, max_lag: int) -> tuple[int, float]:
    """
    Determines the optimal lag 'k' where returns_A[t] leads returns_B[t+k]
    by maximizing the cross-correlation.
    """
    n = min(len(returns_A), len(returns_B))
    best_lag, best_corr = 0, 0.0

    if n < 5: # Not enough data for correlation
        return 0, 0.0

    for k in range(1, max_lag + 1):
        if n - k < 5: # Need at least 5 points for correlation
            break
        
        # Calculate correlation between A's returns and B's future returns
        # r_A[:n-k] corresponds to r_A[t]
        # r_B[k:n] corresponds to r_B[t+k]
        corr_matrix = np.corrcoef(returns_A[:n-k], returns_B[k:n])
        c = float(corr_matrix[0, 1])

        if math.isfinite(c) and c > best_corr:
            best_corr, best_lag = c, k
    return best_lag, best_corr

def _calculate_spread_history(
    prices_A: np.ndarray, prices_B: np.ndarray,
    hedge_ratio: float, optimal_lag: int,
    window: int
) -> np.ndarray:
    """
    Calculates the historical spread for the lookback window
    spread = prices_B[t] - hedge_ratio * prices_A[t - optimal_lag]
    """
    # Ensure there's enough data for the lag and the window
    if optimal_lag >= len(prices_A) or optimal_lag >= len(prices_B) or window > len(prices_A) or window > len(prices_B):
        return np.array([])

    # Align prices such that prices_A[i] leads prices_B[i + optimal_lag]
    # To calculate current_spread = prices_B[current] - hedge_ratio * prices_A[current - optimal_lag]
    # For historical spreads, we need prices_A shifted by `optimal_lag`.
    # So, for prices_B[j], we use prices_A[j - optimal_lag].
    # The historical series will start from index `optimal_lag` for prices_B
    # and index `0` for prices_A.
    
    start_idx_B = optimal_lag
    end_idx_B = len(prices_B) # Up to the last available price
    
    # Ensure we take `window` number of data points for historical calculation
    if (end_idx_B - start_idx_B) < window:
        # Not enough data for the full window with the given lag
        return np.array([])
    
    # Take the last `window` aligned points
    prices_A_lagged = prices_A[end_idx_B - window - optimal_lag : end_idx_B - optimal_lag]
    prices_B_current = prices_B[end_idx_B - window : end_idx_B]

    if len(prices_A_lagged) != window or len(prices_B_current) != window:
        return np.array([]) # Should not happen if logic above is correct, but for safety

    spreads = prices_B_current - hedge_ratio * prices_A_lagged
    return spreads


def _re_evaluate_params(data: MarketData) -> None:
    """
    Re-evaluates cointegration, hedge ratio, and optimal lag for all pairs.
    Updates the global cache `_cached_lead_lag_params`.
    """
    global _cached_lead_lag_params, _last_rebalance_warm_candle_hour, _last_warm_candle_count

    current_warm_candle_count = 0
    if data:
        # Find the max warm candle count to determine if enough data exists
        for pair_data in data.values():
            if pair_data.warm:
                current_warm_candle_count = max(current_warm_candle_count, len(pair_data.warm))
                # Update _last_rebalance_warm_candle_hour to the latest available candle hour
                if _last_rebalance_warm_candle_hour is None or pair_data.warm[-1].hour > _last_rebalance_warm_candle_hour:
                    _last_rebalance_warm_candle_hour = pair_data.warm[-1].hour

    if current_warm_candle_count < MIN_CANDLES_FOR_ANALYSIS:
        logger.debug(f"Not enough warm candles ({current_warm_candle_count}) for analysis. Min required: {MIN_CANDLES_FOR_ANALYSIS}.")
        _cached_lead_lag_params = {} # Clear cache if not enough data
        return

    # Check if rebalance is needed based on warm candle hour or count
    # This ensures re-evaluation happens only when new `REBALANCE_FREQUENCY_BARS` have passed
    # or the warm data has significantly changed.
    if _last_rebalance_warm_candle_hour is not None and \
       current_warm_candle_count - _last_warm_candle_count < REBALANCE_FREQUENCY_BARS:
        return # Not enough new data to rebalance

    logger.info("Re-evaluating lead-lag parameters for all pairs...")
    _cached_lead_lag_params = {} # Clear existing cache for a full re-evaluation
    _last_warm_candle_count = current_warm_candle_count # Update for next rebalance check

    assets = list(data.keys())
    for i, asset_A in enumerate(assets):
        for asset_B in assets:
            if asset_A == asset_B:
                continue

            pd_A = data.get(asset_A)
            pd_B = data.get(asset_B)

            if pd_A is None or pd_B is None or \
               len(pd_A.warm) < MIN_CANDLES_FOR_ANALYSIS or \
               len(pd_B.warm) < MIN_CANDLES_FOR_ANALYSIS:
                continue

            # Use the most recent `MIN_CANDLES_FOR_ANALYSIS` for analysis
            prices_A_full = _get_prices(pd_A.warm[-MIN_CANDLES_FOR_ANALYSIS:])
            prices_B_full = _get_prices(pd_B.warm[-MIN_CANDLES_FOR_ANALYSIS:])

            if len(prices_A_full) < MIN_CANDLES_FOR_ANALYSIS or len(prices_B_full) < MIN_CANDLES_FOR_ANALYSIS:
                continue

            # 1. Calculate optimal lag based on returns (as in original rule)
            returns_A = _get_returns(prices_A_full)
            returns_B = _get_returns(prices_B_full)
            if len(returns_A) < 5 or len(returns_B) < 5:
                continue # Not enough returns for meaningful correlation

            optimal_lag, corr_val = _calculate_optimal_lag_returns(returns_A, returns_B, MAX_LAG_SEARCH_BARS)

            if optimal_lag == 0 or corr_val < MIN_CORRELATION_FOR_PAIR:
                continue # No significant lead-lag relationship found

            # 2. Calculate hedge ratio using OLS on concurrent prices over the lookback window
            # The hedge ratio defines the relationship between current prices.
            # The lag is applied when forming the spread for signal generation.
            prices_A_ols = prices_A_full[-LOOKBACK_WINDOW_BARS:]
            prices_B_ols = prices_B_full[-LOOKBACK_WINDOW_BARS:]
            
            if len(prices_A_ols) < LOOKBACK_WINDOW_BARS or len(prices_B_ols) < LOOKBACK_WINDOW_BARS:
                continue

            hedge_ratio = _calculate_ols_hedge_ratio(prices_A_ols, prices_B_ols)

            if not math.isfinite(hedge_ratio) or hedge_ratio == 0:
                continue # Invalid hedge ratio

            # 3. Spread statistics (proxy for mean-reversion)
            historical_spreads = _calculate_spread_history(
                prices_A_full, prices_B_full,
                hedge_ratio, optimal_lag,
                LOOKBACK_WINDOW_BARS
            )

            if len(historical_spreads) < 10: # Minimum data for mean/std
                continue

            mean_spread = np.mean(historical_spreads)
            std_spread = np.std(historical_spreads)

            if not math.isfinite(mean_spread) or not math.isfinite(std_spread) or std_spread < MIN_SPREAD_STD_DEV:
                logger.debug(f"Invalid spread stats for ({asset_A}, {asset_B}). mean_spread={mean_spread}, std_spread={std_spread}")
                continue

            # If we reach here, we consider the pair valid for trading.
            # We assume 'cointegration' if a stable lead-lag and spread properties are found.
            _cached_lead_lag_params[(asset_A, asset_B)] = {
                'hedge_ratio': hedge_ratio,
                'optimal_lag': optimal_lag,
                'mean_spread': mean_spread,
                'std_spread': std_spread,
                'correlation': corr_val # Store for logging/confidence
            }
            logger.debug(f"Cached params for ({asset_A}, {asset_B}): "
                         f"Hedge Ratio={hedge_ratio:.4f}, Optimal Lag={optimal_lag}, "
                         f"Mean Spread={mean_spread:.4f}, Std Dev Spread={std_spread:.4f}, "
                         f"Correlation={corr_val:.4f}")


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on dynamic cointegration lead-lag arbitrage.
    """
    signals: list[BuySignal | SellSignal] = []
    
    if not data:
        return []

    # Re-evaluate parameters if needed
    _re_evaluate_params(data)

    if not _cached_lead_lag_params:
        logger.debug("No valid lead-lag pairs found or insufficient data after re-evaluation.")
        return []

    seen_targets: set[str] = set()

    for (asset_A, asset_B), params in _cached_lead_lag_params.items():
        if asset_B in seen_targets:
            continue

        pd_A = data.get(asset_A)
        pd_B = data.get(asset_B)

        if pd_A is None or pd_B is None or not pd_B.hot:
            continue

        hedge_ratio = params['hedge_ratio']
        optimal_lag = params['optimal_lag']
        mean_spread = params['mean_spread']
        std_spread = params['std_spread']
        # correlation = params['correlation'] # Not directly used for signal, but for confidence

        # Ensure enough warm candles for the lagged price_A
        # We need pd_A.warm[-1 - optimal_lag] to be valid.
        # This means pd_A.warm must have at least optimal_lag + 1 candles.
        if len(pd_A.warm) < optimal_lag + 1:
            logger.debug(f"Not enough warm candles for {asset_A} to get lagged price (needed {optimal_lag+1}, got {len(pd_A.warm)}).")
            continue

        # Get current prices
        current_tick_B = pd_B.hot[-1]
        current_price_B = current_tick_B.last_price
        timestamp = current_tick_B.polled_at

        # Get the lagged price for asset A
        # pd_A.warm[-1] is the most recent full hour candle.
        # pd_A.warm[-1 - optimal_lag] is the candle 'optimal_lag' hours ago relative to the current full candle.
        lagged_price_A = pd_A.warm[-1 - optimal_lag].close

        # Calculate the current dynamically adjusted spread
        current_spread = current_price_B - hedge_ratio * lagged_price_A

        # Generate signals based on adaptive thresholds
        if std_spread < MIN_SPREAD_STD_DEV:
            logger.debug(f"Std Dev for spread ({asset_A}, {asset_B}) is too small ({std_spread}). Skipping signal.")
            continue

        deviation = (current_spread - mean_spread) / std_spread
        # Scale confidence to [0,1], higher deviation means higher confidence
        confidence = min(1.0, abs(deviation) / ENTRY_THRESHOLD_STD_DEV) 

        if current_spread > mean_spread + ENTRY_THRESHOLD_STD_DEV * std_spread:
            # Spread is too high, meaning B is relatively overvalued compared to lagged A. Expect B to fall.
            signals.append(SellSignal(
                pair=asset_B,
                timestamp=timestamp,
                price=current_price_B,
                confidence=confidence,
                rule_id="e89b2f77-de17-4012-ab8d-23e75a4868dd"
            ))
            seen_targets.add(asset_B)
            logger.info(f"Sell Signal for {asset_B}: Spread too high ({current_spread:.4f} vs mean {mean_spread:.4f}), deviation {deviation:.2f} std dev. Confidence: {confidence:.2f}")

        elif current_spread < mean_spread - ENTRY_THRESHOLD_STD_DEV * std_spread:
            # Spread is too low, meaning B is relatively undervalued compared to lagged A. Expect B to rise.
            signals.append(BuySignal(
                pair=asset_B,
                timestamp=timestamp,
                price=current_price_B,
                confidence=confidence,
                rule_id="e89b2f77-de17-4012-ab8d-23e75a4868dd"
            ))
            seen_targets.add(asset_B)
            logger.info(f"Buy Signal for {asset_B}: Spread too low ({current_spread:.4f} vs mean {mean_spread:.4f}), deviation {deviation:.2f} std dev. Confidence: {confidence:.2f}")

        # Optional: Close positions if spread reverts towards mean
        # This logic would typically be handled by a position manager, but for a signal generator,
        # one could emit a 'close' signal if the deviation is within EXIT_THRESHOLD_STD_DEV.
        # For this rule, we only focus on entry signals as per the prompt.
        # elif abs(current_spread - mean_spread) < EXIT_THRESHOLD_STD_DEV * std_spread:
        #     # This would be a 'close position' signal. Not directly part of Buy/SellSignal output.
        #     # Could be a custom ClosePositionSignal if the platform supports it.
        #     pass

    return signals