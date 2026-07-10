"""Rule 12 — Cross-asset: lead-lag relationship detection, with Volume Confirmation."""
from __future__ import annotations

import math
import numpy as np
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle


MIN_CANDLES = 20  # minimum warm candles per asset for reliable correlation
MAX_LAG = 3  # maximum lead time in hours to consider
CORR_THRESHOLD = 0.5  # minimum Pearson r to treat a lag as a real relationship
LEAD_THRESHOLD = 0.01  # A's k-hour cumulative return must exceed this to signal

# New constants for Volume Confirmation Filter
VOLUME_LOOKBACK_TICKS = 300  # Average 24h volume over the last N ticks (e.g., 300 ticks for 5 minutes at 1 tick/sec)
VOLUME_SURGE_THRESHOLD = 1.25 # Current 24h rolling volume must be 25% higher than its recent average
RULE_ID = "6f505e81-bbc9-46d4-8669-1aa837939163" # Unique ID for this rule version


# ── Pair cache (refreshed when warm tier changes, i.e. ~hourly) ───────────────
_cached_pairs: list[tuple[str, str, int, float]] = []  # (A, B, lag, corr)
_cache_key: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _returns(closes: list[float]) -> np.ndarray:
    arr = np.array(closes, dtype=np.float64)
    return (arr[1:] - arr[:-1]) / np.where(arr[:-1] != 0, arr[:-1], 1.0)


def _best_lag_corr(r_a: np.ndarray, r_b: np.ndarray) -> tuple[int, float]:
    """Return (best_lag, corr) maximising corr(r_A[t], r_B[t+k]) over k."""
    n = min(len(r_a), len(r_b))
    best_lag, best_corr = 0, 0.0
    for k in range(1, MAX_LAG + 1):
        if n - k < 5:  # Need at least 5 data points for correlation
            break
        c = float(np.corrcoef(r_a[: n - k], r_b[k:n])[0, 1])
        if math.isfinite(c) and c > best_corr:
            best_corr, best_lag = c, k
    return best_lag, best_corr


# ── Pair detection (cached) ───────────────────────────────────────────────────


def _detect_pairs(
    asset_returns: dict[str, np.ndarray],
) -> list[tuple[str, str, int, float]]:
    pairs: list[tuple[str, str, int, float]] = []
    assets = list(asset_returns.keys())
    for i, a in enumerate(assets):
        for b in assets:
            if b == a:
                continue
            lag, corr = _best_lag_corr(asset_returns[a], asset_returns[b])
            if corr > CORR_THRESHOLD:
                pairs.append((a, b, lag, corr))
    return pairs


def _get_pairs(data: MarketData) -> list[tuple[str, str, int, float]]:
    global _cached_pairs, _cache_key

    # Cache key: latest warm candle hour for each pair (changes once per hour)
    key = "|".join(
        f"{pair}:{pd.warm[-1].hour.isoformat()}"
        for pair, pd in sorted(data.items())
        if pd.warm
    )
    if key == _cache_key:
        return _cached_pairs

    asset_returns: dict[str, np.ndarray] = {}
    for pair, pd in data.items():
        if len(pd.warm) >= MIN_CANDLES:
            asset_returns[pair] = _returns([c.close for c in pd.warm])

    _cached_pairs = _detect_pairs(asset_returns)
    _cache_key = key

    # Optional: logging for detected pairs
    # if _cached_pairs:
    #     logger_pairs = [(a, b, k, round(c, 2)) for a, b, k, c in _cached_pairs]
    #     import logging
    #     logging.getLogger(__name__).debug("Lead-lag pairs detected: %s", logger_pairs)

    return _cached_pairs


# ── Signal generation ─────────────────────────────────────────────────────────


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    pairs = _get_pairs(data)
    if not pairs:
        return []

    signals: list[BuySignal | SellSignal] = []
    seen_targets: set[str] = set()

    for a, b, lag, corr in pairs:  # a is leader, b is laggard
        if b in seen_targets:
            continue

        pd_a = data.get(a)  # Leader PairData
        pd_b = data.get(b)  # Laggard PairData

        if pd_a is None or pd_b is None:
            continue
        # Ensure enough warm data for leader price change (lag + current candle)
        if len(pd_a.warm) < lag + 1:
            continue
        # Ensure enough hot data for laggard signal and leader volume calculation
        if not pd_b.hot or len(pd_a.hot) < VOLUME_LOOKBACK_TICKS + 1:
            continue

        # 1. Calculate leader_price_change (a_return) over `lag` hours
        closes_a = [c.close for c in pd_a.warm]
        denom_price = closes_a[-lag - 1]
        if denom_price == 0:
            continue
        a_return = (closes_a[-1] - denom_price) / denom_price

        # 2. Calculate leader_volume_ratio for volume confirmation
        current_24h_volume = pd_a.hot[-1].volume_24h

        # Calculate average 24h volume over the lookback period from past ticks
        # We need VOLUME_LOOKBACK_TICKS past ticks to average over.
        # So, we check `len(pd_a.hot) < VOLUME_LOOKBACK_TICKS + 1` above.
        # This slice `[-VOLUME_LOOKBACK_TICKS - 1 : -1]` gets the last `VOLUME_LOOKBACK_TICKS`
        # ticks, *excluding* the very latest one (`pd_a.hot[-1]`) for the average.
        past_volumes_24h = [t.volume_24h for t in pd_a.hot[-VOLUME_LOOKBACK_TICKS - 1 : -1]]

        if not past_volumes_24h:
            # This case should ideally be caught by the len(pd_a.hot) check,
            # but as a safeguard, if list is empty, skip volume check.
            continue
        
        avg_24h_volume = np.mean(past_volumes_24h)

        leader_volume_ratio: float
        if avg_24h_volume == 0:
            # If historical average volume is zero, any positive current volume implies a surge.
            leader_volume_ratio = float('inf') if current_24h_volume > 0 else 0.0
        else:
            leader_volume_ratio = current_24h_volume / avg_24h_volume
        
        ts = pd_b.hot[-1].polled_at
        price = pd_b.hot[-1].last_price

        # 3. Emit Buy signal for the lagging asset if conditions met
        if (a_return > LEAD_THRESHOLD and
            leader_volume_ratio > VOLUME_SURGE_THRESHOLD):
            seen_targets.add(b)
            signals.append(BuySignal(pair=b, timestamp=ts, price=price, confidence=corr, rule_id=RULE_ID))
        # 4. Emit Sell signal for the lagging asset if conditions met
        elif (a_return < -LEAD_THRESHOLD and
              leader_volume_ratio > VOLUME_SURGE_THRESHOLD):
            seen_targets.add(b)
            signals.append(SellSignal(pair=b, timestamp=ts, price=price, confidence=corr, rule_id=RULE_ID))

    return signals