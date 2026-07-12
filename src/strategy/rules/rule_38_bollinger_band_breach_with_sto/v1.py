"""Rule — Bollinger Band Breach with Stochastic Divergence and Adaptive Volume (d5321e15)."""
from __future__ import annotations

import numpy as np

from src.agent.models import BuySignal, MarketData, SellSignal

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BB_PERIOD: int = 20
BB_STD_DEV: float = 2.0
STOCH_K_PERIOD: int = 14
STOCH_D_PERIOD: int = 3
VOLUME_AVG_PERIOD: int = 20
VOLUME_MULTIPLIER: float = 1.5
# Number of bars separating the two reference points used in divergence detection
DIVERGENCE_LOOKBACK: int = 5

# Minimum warm candles required to compute every indicator
MIN_CANDLES: int = max(
    BB_PERIOD,
    STOCH_K_PERIOD + STOCH_D_PERIOD - 1,
    VOLUME_AVG_PERIOD + 1,
)

RULE_ID: str = "d5321e15-32c9-4b6f-b13d-648b04a6a709"


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------


def _bollinger_bands(closes: np.ndarray) -> tuple[float, float, float]:
    """Return (lower_band, sma, upper_band) over the last BB_PERIOD bars."""
    window = closes[-BB_PERIOD:]
    sma = float(np.mean(window))
    std = float(np.std(window, ddof=0))
    return sma - BB_STD_DEV * std, sma, sma + BB_STD_DEV * std


def _stochastic_k(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> np.ndarray:
    """
    Compute raw Stochastic %K for every bar from index STOCH_K_PERIOD-1 onward.
    Result length = len(closes) - STOCH_K_PERIOD + 1.
    """
    n = len(closes)
    k: list[float] = []
    for i in range(STOCH_K_PERIOD - 1, n):
        hi = float(np.max(highs[i - STOCH_K_PERIOD + 1 : i + 1]))
        lo = float(np.min(lows[i - STOCH_K_PERIOD + 1 : i + 1]))
        rng = hi - lo
        k.append(50.0 if rng == 0.0 else 100.0 * (closes[i] - lo) / rng)
    return np.array(k, dtype=float)


def _stochastic_d(k_values: np.ndarray) -> np.ndarray:
    """Smooth %K with a simple moving average of STOCH_D_PERIOD to yield %D."""
    out: list[float] = []
    for i in range(STOCH_D_PERIOD - 1, len(k_values)):
        out.append(float(np.mean(k_values[i - STOCH_D_PERIOD + 1 : i + 1])))
    return np.array(out, dtype=float)


def _bullish_divergence(closes_aligned: np.ndarray, k_values: np.ndarray) -> bool:
    """
    Bullish divergence: price makes a lower level while %K makes a higher level
    relative to the bar DIVERGENCE_LOOKBACK bars ago.  This indicates that
    bearish momentum is waning despite the new price low — a classic mean-
    reversion precursor.
    """
    if len(closes_aligned) <= DIVERGENCE_LOOKBACK or len(k_values) <= DIVERGENCE_LOOKBACK:
        return False
    price_lower = closes_aligned[-1] < closes_aligned[-DIVERGENCE_LOOKBACK - 1]
    stoch_higher = k_values[-1] > k_values[-DIVERGENCE_LOOKBACK - 1]
    return bool(price_lower and stoch_higher)


def _bearish_divergence(closes_aligned: np.ndarray, k_values: np.ndarray) -> bool:
    """
    Bearish divergence: price makes a higher level while %K makes a lower level,
    signalling that bullish momentum is waning at the price extreme.
    """
    if len(closes_aligned) <= DIVERGENCE_LOOKBACK or len(k_values) <= DIVERGENCE_LOOKBACK:
        return False
    price_higher = closes_aligned[-1] > closes_aligned[-DIVERGENCE_LOOKBACK - 1]
    stoch_lower = k_values[-1] < k_values[-DIVERGENCE_LOOKBACK - 1]
    return bool(price_higher and stoch_lower)


def _adaptive_high_volume(volumes: np.ndarray) -> bool:
    """
    Return True when the latest bar's volume exceeds the rolling mean of the
    preceding VOLUME_AVG_PERIOD bars by at least VOLUME_MULTIPLIER.
    The baseline deliberately excludes the current bar to avoid look-ahead.
    """
    if len(volumes) < VOLUME_AVG_PERIOD + 1:
        return False
    baseline = float(np.mean(volumes[-(VOLUME_AVG_PERIOD + 1) : -1]))
    if baseline <= 0.0:
        return False
    return float(volumes[-1]) > baseline * VOLUME_MULTIPLIER


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Emit Buy/Sell signals when all three conditions hold simultaneously:

    Buy:
      1. Close < Lower Bollinger Band  (price at an extreme low)
      2. Bullish Stochastic divergence  (price lower low, %K higher low)
      3. Adaptive volume surge          (current volume > avg × multiplier)

    Sell:
      1. Close > Upper Bollinger Band  (price at an extreme high)
      2. Bearish Stochastic divergence  (price higher high, %K lower high)
      3. Adaptive volume surge          (current volume > avg × multiplier)
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm
        if len(candles) < MIN_CANDLES:
            continue

        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        # --- 1. Bollinger Band breach ---
        lower_bb, _sma, upper_bb = _bollinger_bands(closes)
        current_close = closes[-1]
        breach_lower = current_close < lower_bb
        breach_upper = current_close > upper_bb

        if not (breach_lower or breach_upper):
            continue

        # --- 2. Stochastic %K series ---
        k_series = _stochastic_k(highs, lows, closes)
        # closes_aligned[i] is the closing price corresponding to k_series[i]
        closes_aligned = closes[STOCH_K_PERIOD - 1 :]

        if len(k_series) <= DIVERGENCE_LOOKBACK:
            continue

        # --- 3. Adaptive volume confirmation ---
        if not _adaptive_high_volume(volumes):
            continue

        # Prefer the freshest hot tick for timestamp / price; fall back to candle
        if pair_data.hot:
            latest_tick = pair_data.hot[-1]
            ts = latest_tick.polled_at
            price = latest_tick.last_price
        else:
            ts = candles[-1].hour
            price = current_close

        # --- Buy: lower BB breach + bullish Stochastic divergence ---
        if breach_lower and _bullish_divergence(closes_aligned, k_series):
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=ts,
                    price=price,
                    rule_id=RULE_ID,
                    confidence=0.77,
                )
            )

        # --- Sell: upper BB breach + bearish Stochastic divergence ---
        elif breach_upper and _bearish_divergence(closes_aligned, k_series):
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=ts,
                    price=price,
                    rule_id=RULE_ID,
                    confidence=0.77,
                )
            )

    return signals