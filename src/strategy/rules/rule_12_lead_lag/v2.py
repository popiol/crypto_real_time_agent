"""Rule 12 — Cross-asset: lead-lag relationship detection with volatility filter (v2).

This modification to rule_12_lead_lag_v1 introduces a volatility filter for the lagging asset.
A Buy or Sell signal will only be generated if the leading asset's movement meets the existing
criteria AND the lagging asset's short-term historical price volatility (standard deviation of returns)
is below a predefined threshold. This aims to reduce false signals during periods of high uncertainty
or erratic price action in the lagging asset, where the lead-lag relationship might be less reliable.

For every ordered pair of assets (A, B), computes the Pearson cross-correlation
of their hourly returns at lags k = 1 … MAX_LAG using the warm tier:

    corr(r_A[t], r_B[t + k])   for k = 1, 2, …, MAX_LAG

A positive correlation > CORR_THRESHOLD at lag k means A's return today
predicts B's return k hours from now — A is the leader, B is the follower.

Buy signal:  A's k-hour cumulative return >  LEAD_THRESHOLD → B expected to rise.
Sell signal: A's k-hour cumulative return < -LEAD_THRESHOLD → B expected to fall.

Signals are filtered:
A signal is only generated if the lagging asset's `VOLATILITY_WINDOW`-hour
return volatility (standard deviation of hourly returns) is below `VOLATILITY_THRESHOLD`.

At most one signal is emitted per target asset per cycle.

Detected pairs are cached until the warm tier changes (once per hour) so the
O(N²) correlation scan runs at most once per warm refresh, not every second.
"""

from __future__ import annotations

import math
import logging

import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


# --- Rule Constants ---
MIN_CANDLES = 20  # minimum warm candles per asset for reliable correlation and volatility
MAX_LAG = 3  # maximum lead time in hours to consider
CORR_THRESHOLD = 0.5  # minimum Pearson r to treat a lag as a real relationship
LEAD_THRESHOLD = 0.01  # A's k-hour cumulative return must exceed this to signal

VOLATILITY_WINDOW = 5 # Number of warm candles (hours) to calculate lagging asset volatility
VOLATILITY_THRESHOLD = 0.005 # Lagging asset's hourly return std dev must be below this to signal (e.g., 0.5%)
                             # Note: VOLATILITY_WINDOW must be at least 2 for _returns to produce data,
                             # and ideally >=3 for a meaningful standard deviation. Current 5 is fine.

RULE_ID = "rule_12_lead_lag_v2" # Unique identifier for this rule

# --- Pair cache (refreshed when warm tier changes, i.e. ~hourly) ---
_cached_pairs: list[tuple[str, str, int, float]] = []  # (A, B, lag, corr)
_cache_key: str = ""


# --- Helpers ---


def _returns(closes: list[float]) -> np.ndarray:
    """Calculates period-over-period returns from a list of closing prices."""
    arr = np.array(closes, dtype=np.float64)
    # Prevent division by zero if a previous price was zero.
    # If arr[:-1] contains zero, the corresponding return will be 0.0, which is
    # a graceful handling for numerical stability, though it implies data issues.
    return (arr[1:] - arr[:-1]) / np.where(arr[:-1] != 0, arr[:-1], 1.0)


def _best_lag_corr(r_a: np.ndarray, r_b: np.ndarray) -> tuple[int, float]:
    """
    Return (best_lag, corr) maximising corr(r_A[t], r_B[t+k]) over k.
    Only considers positive correlations.
    """
    n = min(len(r_a), len(r_b))
    best_lag, best_corr = 0, 0.0
    for k in range(1, MAX_LAG + 1):
        if n - k < 5: # Need at least 5 data points for meaningful correlation calculation.
            break
        # Pearson correlation coefficient between r_A[t] and r_B[t+k]
        # np.corrcoef returns a 2x2 matrix. We need the off-diagonal element.
        c = float(np.corrcoef(r_a[: n - k], r_b[k:n])[0, 1])
        if math.isfinite(c) and c > best_corr:
            best_corr, best_lag = c, k
    return best_lag, best_corr


# --- Pair detection (cached) ---


def _detect_pairs(
    asset_returns: dict[str, np.ndarray],
) -> list[tuple[str, str, int, float]]:
    """
    Detects lead-lag relationships between all pairs of assets based on returns
    and correlation.
    """
    pairs: list[tuple[str, str, int, float]] = []
    assets = list(asset_returns.keys())
    for i, a in enumerate(assets):
        for b in assets:
            if b == a:
                continue # An asset cannot lead itself
            lag, corr = _best_lag_corr(asset_returns[a], asset_returns[b])
            if corr > CORR_THRESHOLD:
                pairs.append((a, b, lag, corr))
    return pairs


def _get_pairs(data: MarketData) -> list[tuple[str, str, int, float]]:
    """
    Retrieves or detects lead-lag pairs. Caches results to avoid re-computation
    until warm tier data changes.
    """
    global _cached_pairs, _cache_key

    # Cache key: latest warm candle hour for each pair.
    # This ensures the cache is refreshed approximately once per hour when new warm data arrives.
    key = "|".join(
        f"{pair}:{pd.warm[-1].hour.isoformat()}"
        for pair, pd in sorted(data.items())
        if pd.warm
    )
    if key == _cache_key:
        return _cached_pairs

    asset_returns: dict[str, np.ndarray] = {}
    for pair, pd in data.items():
        # Ensure enough warm candles for correlation calculation and volatility calculation.
        # MIN_CANDLES is set to 20, which is sufficient for MAX_LAG (3) and VOLATILITY_WINDOW (5).
        if len(pd.warm) >= MIN_CANDLES:
            asset_returns[pair] = _returns([c.close for c in pd.warm])

    _cached_pairs = _detect_pairs(asset_returns)
    _cache_key = key

    if _cached_pairs:
        logger = logging.getLogger(__name__)
        logger_pairs = [(a, b, k, round(c, 2)) for a, b, k, c in _cached_pairs]
        logger.debug("Lead-lag pairs detected: %s", logger_pairs)

    return _cached_pairs


# --- Signal Generation ---


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on detected lead-lag relationships,
    filtered by lagging asset volatility.
    """
    pairs = _get_pairs(data)
    if not pairs:
        return []

    signals: list[BuySignal | SellSignal] = []
    seen_targets: set[str] = set() # To ensure only one signal per target asset (B) per cycle

    for a, b, lag, corr in pairs:
        if b in seen_targets:
            continue

        pd_a = data.get(a)
        pd_b = data.get(b)
        if pd_a is None or pd_b is None:
            continue

        # Check for sufficient warm candle data for leading asset (A) return calculation
        # We need `lag + 1` warm candles to calculate the `lag`-period return for asset A.
        # Example: if lag=1, we need warm[-2] and warm[-1]. So length must be at least 2.
        if len(pd_a.warm) < lag + 1:
            continue

        # Check for sufficient warm candle data for lagging asset (B) volatility calculation
        # We need `VOLATILITY_WINDOW` warm candles for asset B.
        if len(pd_b.warm) < VOLATILITY_WINDOW:
            continue
        
        # We also need at least one hot tick for the target asset (B) to get current price and timestamp
        if not pd_b.hot:
            continue

        # Calculate leading asset's k-hour cumulative return (return over `lag` periods)
        closes_a = [c.close for c in pd_a.warm]
        denom = closes_a[-lag - 1] # Price `lag` hours ago
        if denom == 0: # Avoid division by zero for return calculation
            continue
        a_return = (closes_a[-1] - denom) / denom # Current price minus price `lag` hours ago

        # Calculate lagging asset's short-term historical price volatility
        # Use the most recent `VOLATILITY_WINDOW` warm candles for asset B.
        closes_b_volatility_window = [c.close for c in pd_b.warm[-VOLATILITY_WINDOW:]]
        returns_b_volatility_window = _returns(closes_b_volatility_window)
        
        # If _returns produced an empty array (e.g., only one price in window), skip
        if len(returns_b_volatility_window) == 0:
            continue

        # Calculate standard deviation of returns for the lagging asset
        lagging_asset_volatility = np.std(returns_b_volatility_window)

        # Get current timestamp and price for the lagging asset (B) for signal generation
        ts = pd_b.hot[-1].polled_at
        price = pd_b.hot[-1].last_price

        # Apply volatility filter: only generate a signal if lagging asset's volatility is low
        if lagging_asset_volatility < VOLATILITY_THRESHOLD:
            # Apply original lead-lag signal logic
            if a_return > LEAD_THRESHOLD:
                seen_targets.add(b)
                signals.append(
                    BuySignal(
                        pair=b,
                        timestamp=ts,
                        price=price,
                        confidence=corr,
                        rule_id=RULE_ID # Set the unique rule ID
                    )
                )
            elif a_return < -LEAD_THRESHOLD:
                seen_targets.add(b)
                signals.append(
                    SellSignal(
                        pair=b,
                        timestamp=ts,
                        price=price,
                        confidence=corr,
                        rule_id=RULE_ID # Set the unique rule ID
                    )
                )

    return signals