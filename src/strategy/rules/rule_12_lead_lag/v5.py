from __future__ import annotations

import math
import logging
from datetime import datetime

import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick

logger = logging.getLogger(__name__)

# ── Configuration Parameters ──────────────────────────────────────────────────
# These parameters refine the lead-lag rule by introducing dynamic lag,
# stricter correlation, and momentum confirmation.

CORRELATION_WINDOW_SIZE = 24  # Number of warm candles (hours) for correlation analysis.
MAX_LAG = 3  # Maximum lead time in hours to consider for lead-lag relationship.
CORRELATION_SIGNIFICANCE_THRESHOLD = 0.6  # Minimum absolute Pearson r to consider a lead-lag relationship significant.
LEADING_ASSET_MOVEMENT_THRESHOLD = 0.005  # Leading asset's k-hour cumulative return must exceed this for a signal.
MOMENTUM_WINDOW_SIZE = 3  # Number of warm candles (hours) for lagging asset's Rate of Change (ROC).
MOMENTUM_CONFIRMATION_THRESHOLD = 0.001  # Minimum absolute ROC (percentage) for lagging asset to confirm direction (e.g., 0.1% change).

# ── Pair cache (refreshed when warm tier changes, i.e. ~hourly) ───────────────
_cached_pairs: list[tuple[str, str, int, float]] = []  # (leader_asset, follower_asset, optimal_lag, max_correlation)
_cache_key: str = ""

# ── Helper Functions ──────────────────────────────────────────────────────────

def _get_closes(candles: list[WarmCandle]) -> np.ndarray:
    """Extract closing prices from a list of WarmCandle objects."""
    return np.array([c.close for c in candles], dtype=np.float64)

def _returns(closes: np.ndarray) -> np.ndarray:
    """Calculate percentage returns from a series of closing prices."""
    if len(closes) < 2:
        return np.array([])
    # Avoid division by zero by replacing zero denominators with 1.0 (results in 0 return for that step)
    return (closes[1:] - closes[:-1]) / np.where(closes[:-1] != 0, closes[:-1], 1.0)

def _calculate_roc(prices: np.ndarray, window: int) -> float | None:
    """
    Calculate the Rate of Change (ROC) for the latest period.
    ROC = ((Close_t - Close_{t-window}) / Close_{t-window})
    Returns None if insufficient data or division by zero.
    """
    if len(prices) < window + 1:
        return None
    current_price = prices[-1]
    price_window_ago = prices[-window - 1] # Index for the price 'window' periods ago
    if price_window_ago == 0:
        return None
    return (current_price - price_window_ago) / price_window_ago

def _best_lag_corr(r_a: np.ndarray, r_b: np.ndarray) -> tuple[int, float]:
    """
    Return (optimal_lag_period, max_correlation_strength) maximizing
    the absolute correlation between r_A[t] and r_B[t+k] over k.
    """
    n = min(len(r_a), len(r_b))
    best_lag, best_corr = 0, 0.0
    
    # Minimum data points required for a statistically meaningful correlation
    min_data_points_for_corr = 5 
    
    for k in range(1, MAX_LAG + 1):
        if n - k < min_data_points_for_corr:
            # Not enough overlapping data points to calculate correlation reliably
            continue
        
        # Calculate Pearson correlation coefficient
        # r_a[:n-k] aligns with r_b[k:n] for cross-correlation at lag k
        corr_matrix = np.corrcoef(r_a[: n - k], r_b[k:n])
        c = float(corr_matrix[0, 1]) if corr_matrix.shape == (2,2) else np.nan

        if math.isfinite(c) and abs(c) > abs(best_corr):
            best_corr, best_lag = c, k
    return best_lag, best_corr


# ── Pair Detection (cached) ───────────────────────────────────────────────────

def _detect_pairs(
    asset_returns: dict[str, np.ndarray],
) -> list[tuple[str, str, int, float]]:
    """
    Detects lead-lag pairs based on cross-correlation, dynamic lag,
    and a significance threshold.
    """
    pairs: list[tuple[str, str, int, float]] = []
    assets = list(asset_returns.keys())

    for i, a in enumerate(assets):
        for b in assets:
            if b == a:
                continue # An asset cannot lead itself

            # Ensure enough return data for both assets for the correlation window
            # _returns function reduces length by 1, so need CORRELATION_WINDOW_SIZE returns.
            if len(asset_returns[a]) < CORRELATION_WINDOW_SIZE or \
               len(asset_returns[b]) < CORRELATION_WINDOW_SIZE:
                continue

            optimal_lag, max_correlation = _best_lag_corr(asset_returns[a], asset_returns[b])

            # Check if correlation strength is significant (absolute value)
            if abs(max_correlation) >= CORRELATION_SIGNIFICANCE_THRESHOLD:
                pairs.append((a, b, optimal_lag, max_correlation))
    return pairs

def _get_pairs(data: MarketData) -> list[tuple[str, str, int, float]]:
    """
    Retrieves lead-lag pairs from cache or re-detects them if market data
    (warm tier) has changed. The cache key is based on the latest hourly candle.
    """
    global _cached_pairs, _cache_key

    # Generate a cache key based on the latest warm candle hour for all available pairs.
    # This ensures the cache is refreshed approximately once per hour.
    key_parts = []
    for pair_name, pd in sorted(data.items()):
        if pd.warm:
            key_parts.append(f"{pair_name}:{pd.warm[-1].hour.isoformat()}")
    key = "|".join(key_parts)

    if key == _cache_key and _cached_pairs: # Return cached if key matches and cache is not empty
        return _cached_pairs

    asset_returns: dict[str, np.ndarray] = {}
    for pair_name, pd in data.items():
        # Need CORRELATION_WINDOW_SIZE + 1 candles to produce CORRELATION_WINDOW_SIZE returns
        if len(pd.warm) >= CORRELATION_WINDOW_SIZE + 1:
            # Use the last CORRELATION_WINDOW_SIZE + 1 candles to calculate returns
            asset_returns[pair_name] = _returns(_get_closes(pd.warm[-(CORRELATION_WINDOW_SIZE + 1):]))
        else:
            logger.debug(
                f"Insufficient warm data for {pair_name} ({len(pd.warm)} candles) "
                f"for correlation window {CORRELATION_WINDOW_SIZE}."
            )

    _cached_pairs = _detect_pairs(asset_returns)
    _cache_key = key

    if _cached_pairs:
        logger.debug("Lead-lag pairs detected: %s", [(a, b, k, round(c, 2)) for a, b, k, c in _cached_pairs])
    else:
        logger.debug("No significant lead-lag pairs detected.")

    return _cached_pairs


# ── Signal Generation ─────────────────────────────────────────────────────────

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on adaptive lead-lag relationships
    and lagging asset momentum confirmation.
    """
    pairs = _get_pairs(data)
    if not pairs:
        return []

    signals: list[BuySignal | SellSignal] = []
    seen_targets: set[str] = set() # Ensure only one signal per target asset per cycle

    for leader_asset, follower_asset, optimal_lag, correlation_strength in pairs:
        if follower_asset in seen_targets:
            continue

        pd_leader = data.get(leader_asset)
        pd_follower = data.get(follower_asset)

        if pd_leader is None or pd_follower is None:
            logger.debug(f"Missing PairData for leader {leader_asset} or follower {follower_asset}.")
            continue

        # Data sufficiency checks for signal generation
        # Leader asset needs enough warm candles to calculate movement over optimal_lag
        # (optimal_lag + 1 candles for a movement over optimal_lag periods)
        if len(pd_leader.warm) < optimal_lag + 1:
            logger.debug(
                f"Insufficient warm data for leader {leader_asset} ({len(pd_leader.warm)} candles) "
                f"for lag {optimal_lag}."
            )
            continue

        # Follower asset needs enough warm candles for momentum calculation
        # (MOMENTUM_WINDOW_SIZE + 1 candles for ROC over MOMENTUM_WINDOW_SIZE periods)
        if len(pd_follower.warm) < MOMENTUM_WINDOW_SIZE + 1:
            logger.debug(
                f"Insufficient warm data for follower {follower_asset} ({len(pd_follower.warm)} candles) "
                f"for momentum window {MOMENTUM_WINDOW_SIZE}."
            )
            continue

        # Follower asset needs hot data for current price and timestamp
        if not pd_follower.hot:
            logger.debug(f"No hot data for follower {follower_asset}.")
            continue

        # 1. Calculate leading asset's movement over the optimal_lag_period
        leader_closes = _get_closes(pd_leader.warm)
        # The movement is from `optimal_lag` hours ago to the present
        denom_leader = leader_closes[-optimal_lag - 1]
        if denom_leader == 0:
            logger.debug(f"Leader asset {leader_asset} price {optimal_lag} hours ago was zero.")
            continue
        
        leader_movement_pct = (leader_closes[-1] - denom_leader) / denom_leader

        # If leader's movement is not significant, no signal
        if abs(leader_movement_pct) < LEADING_ASSET_MOVEMENT_THRESHOLD:
            continue

        # 2. Calculate lagging asset's short-term momentum (ROC)
        follower_closes = _get_closes(pd_follower.warm)
        follower_roc = _calculate_roc(follower_closes, MOMENTUM_WINDOW_SIZE)

        if follower_roc is None:
            logger.debug(f"Could not calculate ROC for follower {follower_asset}.")
            continue

        # Get current market data for signal
        ts = pd_follower.hot[-1].polled_at
        price = pd_follower.hot[-1].last_price
        
        # 3. Apply signal generation logic based on correlation direction,
        #    leading asset movement, and lagging asset momentum confirmation.
        
        # Case A: Positive correlation (Leader UP -> Follower UP, Leader DOWN -> Follower DOWN)
        if correlation_strength > 0:
            if leader_movement_pct > LEADING_ASSET_MOVEMENT_THRESHOLD: # Bullish lead
                if follower_roc > MOMENTUM_CONFIRMATION_THRESHOLD: # Follower momentum confirms UP
                    signals.append(BuySignal(
                        pair=follower_asset,
                        timestamp=ts,
                        price=price,
                        confidence=abs(correlation_strength),
                        rule_id="adaptive_lead_lag_momentum_conf"
                    ))
                    seen_targets.add(follower_asset)
            elif leader_movement_pct < -LEADING_ASSET_MOVEMENT_THRESHOLD: # Bearish lead
                if follower_roc < -MOMENTUM_CONFIRMATION_THRESHOLD: # Follower momentum confirms DOWN
                    signals.append(SellSignal(
                        pair=follower_asset,
                        timestamp=ts,
                        price=price,
                        confidence=abs(correlation_strength),
                        rule_id="adaptive_lead_lag_momentum_conf"
                    ))
                    seen_targets.add(follower_asset)
        
        # Case B: Negative correlation (Leader UP -> Follower DOWN, Leader DOWN -> Follower UP)
        elif correlation_strength < 0:
            if leader_movement_pct > LEADING_ASSET_MOVEMENT_THRESHOLD: # Bullish lead
                if follower_roc < -MOMENTUM_CONFIRMATION_THRESHOLD: # Follower momentum confirms DOWN (inverse)
                    signals.append(SellSignal(
                        pair=follower_asset,
                        timestamp=ts,
                        price=price,
                        confidence=abs(correlation_strength),
                        rule_id="adaptive_lead_lag_momentum_conf"
                    ))
                    seen_targets.add(follower_asset)
            elif leader_movement_pct < -LEADING_ASSET_MOVEMENT_THRESHOLD: # Bearish lead
                if follower_roc > MOMENTUM_CONFIRMATION_THRESHOLD: # Follower momentum confirms UP (inverse)
                    signals.append(BuySignal(
                        pair=follower_asset,
                        timestamp=ts,
                        price=price,
                        confidence=abs(correlation_strength),
                        rule_id="adaptive_lead_lag_momentum_conf"
                    ))
                    seen_targets.add(follower_asset)

    return signals