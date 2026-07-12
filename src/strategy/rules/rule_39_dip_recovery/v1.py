"""Rule 39 — Dip & recovery: large price drops and subsequent recoveries.

Buy signal:  current price has fallen more than DROP_THRESHOLD below the
             last candle's high, and the 5-hour change confirms the same
             downward direction as the 1-hour change.

Sell signal: current price has recovered more than RECOVERY_THRESHOLD above
             the last candle's low, provided the intra-candle dip was at
             least MIN_DROP_FOR_RECOVERY deep, and the 5-hour change
             confirms the same upward direction as the 1-hour change.
"""

from __future__ import annotations

from src.agent.models import BuySignal, MarketData, SellSignal

DROP_THRESHOLD: float = 0.01  # ≥5 % below recent high → buy
RECOVERY_THRESHOLD: float = 0.005  # ≥3 % above recent low → sell
MIN_DROP_FOR_RECOVERY: float = 0.01  # dip must have been ≥4 % to qualify for sell
MIN_WARM_CANDLES: int = 6  # need at least 6 hourly candles for reliable extremes
TREND_WINDOW: int = 5  # hours over which the trend must agree with the last-hour move


def _close_change(warm: list, offset: int) -> float:
    """Fractional change of warm[-1].close relative to warm[-1 - offset].close."""
    base = warm[-1 - offset].close
    return (warm[-1].close - base) / base if base != 0 else 0.0


def _trend_confirms(warm: list) -> bool:
    """Return True when the 1-hour and 5-hour changes differ by less than 1 %."""
    change_1h = _close_change(warm, 1)
    change_5h = _close_change(warm, TREND_WINDOW)
    return abs(change_1h - change_5h) < 0.01


def _dip_depth(warm: list) -> tuple[float, float]:
    """Return (depth, low_after_high) where low is taken from candles after the 24h peak."""
    high_idx = max(range(len(warm)), key=lambda i, w=warm: w[i].high)
    high_24h = warm[high_idx].high
    candles_after = warm[high_idx + 1 :]
    if not candles_after or high_24h == 0:
        return 0.0, 0.0
    low_24h = min(c.low for c in candles_after)
    return (high_24h - low_24h) / high_24h, low_24h


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pd in data.items():
        if not pd.hot or len(pd.warm) < MIN_WARM_CANDLES:
            continue

        warm = pd.warm
        last_candle = warm[-1]
        recent_high = last_candle.high

        if recent_high == 0:
            continue

        if not _trend_confirms(warm):
            continue

        tick = pd.hot[-1]
        price = tick.last_price
        ts = tick.polled_at

        # Buy: price deeply below recent high
        drop_from_high = (recent_high - price) / recent_high
        if drop_from_high > DROP_THRESHOLD:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=ts,
                    price=price,
                    confidence=min(drop_from_high, 1.0),
                )
            )
            continue

        # Sell: price has recovered from a genuine dip
        dip_depth, low_24h = _dip_depth(warm)
        if low_24h == 0:
            continue
        recovery = (price - low_24h) / low_24h
        if dip_depth > MIN_DROP_FOR_RECOVERY and recovery > RECOVERY_THRESHOLD:
            signals.append(
                SellSignal(
                    pair=pair, timestamp=ts, price=price, confidence=min(recovery, 1.0)
                )
            )

    return signals
