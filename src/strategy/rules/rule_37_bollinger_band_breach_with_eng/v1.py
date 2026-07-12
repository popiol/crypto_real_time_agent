"""Rule 00a2129e — Bollinger Band Breach with Engulfing Candlestick Reversal and MFI/Volume Confirmation."""
from __future__ import annotations

import math
import statistics
from typing import NamedTuple

from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# ── Parameters ────────────────────────────────────────────────────────────────
BB_PERIOD: int = 20
BB_STD_DEV: float = 2.0
MFI_PERIOD: int = 14
MFI_OVERSOLD: float = 20.0
MFI_OVERBOUGHT: float = 80.0
VOLUME_MA_PERIOD: int = 20
MIN_VOLUME_MULTIPLIER: float = 1.2

# We need BB values at both P (index -2) and C (index -1), which requires
# BB_PERIOD candles before P, so BB_PERIOD + 2 total.
# MFI requires MFI_PERIOD + 1 candles for one value and MFI_PERIOD + 2 for two
# consecutive values (mfi_0 and mfi_1).
# Volume MA requires VOLUME_MA_PERIOD candles.
MIN_CANDLES: int = max(BB_PERIOD + 2, MFI_PERIOD + 2, VOLUME_MA_PERIOD + 1)

RULE_ID: str = "00a2129e-13a5-4071-a219-4ad164b2ae9c"
CONFIDENCE: float = 0.81


# ── Indicator helpers ─────────────────────────────────────────────────────────

class BollingerBands(NamedTuple):
    upper: float
    middle: float
    lower: float


def _bollinger_bands(closes: list[float]) -> BollingerBands:
    """Compute Bollinger Bands from the last BB_PERIOD closing prices.

    Uses sample standard deviation (ddof=1), consistent with most trading
    platforms (e.g. TradingView).
    """
    window = closes[-BB_PERIOD:]
    mid = statistics.mean(window)
    std = statistics.stdev(window)  # sample stdev (n-1 denominator)
    return BollingerBands(
        upper=mid + BB_STD_DEV * std,
        middle=mid,
        lower=mid - BB_STD_DEV * std,
    )


def _typical_price(candle: WarmCandle) -> float:
    return (candle.high + candle.low + candle.close) / 3.0


def _compute_mfi(candles: list[WarmCandle]) -> float:
    """Compute Money Flow Index over the last MFI_PERIOD candles.

    Requires MFI_PERIOD + 1 candles (the extra one establishes the prior
    typical price needed to determine money-flow direction on the first bar).

    Returns NaN if there is insufficient data or if negative money flow is zero
    and positive is also zero (flat market).
    """
    window = candles[-(MFI_PERIOD + 1):]
    if len(window) < MFI_PERIOD + 1:
        return math.nan

    positive_mf = 0.0
    negative_mf = 0.0

    for i in range(1, len(window)):
        prev_tp = _typical_price(window[i - 1])
        curr_tp = _typical_price(window[i])
        raw_mf = curr_tp * window[i].volume

        if curr_tp > prev_tp:
            positive_mf += raw_mf
        elif curr_tp < prev_tp:
            negative_mf += raw_mf
        # Equal typical prices contribute to neither flow bucket.

    if negative_mf == 0.0:
        # Avoid division by zero: all flow is positive → MFI = 100.
        return 100.0 if positive_mf > 0.0 else math.nan

    mfr = positive_mf / negative_mf
    return 100.0 - (100.0 / (1.0 + mfr))


# ── Signal logic ──────────────────────────────────────────────────────────────

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """Bollinger Band breach + engulfing candlestick reversal + MFI/volume confirmation."""
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm
        if len(candles) < MIN_CANDLES:
            continue

        c = candles[-1]   # current candle
        p = candles[-2]   # previous candle

        closes = [candle.close for candle in candles]

        # ── Bollinger Bands ──────────────────────────────────────────────────
        # BB at P: computed using closes up to and including P (i.e. closes[:-1])
        closes_at_p = closes[:-1]
        if len(closes_at_p) < BB_PERIOD:
            continue
        bb_p = _bollinger_bands(closes_at_p)

        # BB at C: computed using all closes including C
        bb_c = _bollinger_bands(closes)

        # ── MFI (current and one bar prior) ──────────────────────────────────
        # mfi_0: MFI at C, computed over candles ending at index -1
        mfi_0 = _compute_mfi(candles)
        # mfi_1: MFI at P, computed over candles ending at index -2
        mfi_1 = _compute_mfi(candles[:-1])

        if math.isnan(mfi_0) or math.isnan(mfi_1):
            continue

        # ── Volume moving average ─────────────────────────────────────────────
        volume_ma = statistics.mean(
            candle.volume for candle in candles[-VOLUME_MA_PERIOD:]
        )
        volume_threshold = volume_ma * MIN_VOLUME_MULTIPLIER

        # ── Candlestick character ─────────────────────────────────────────────
        p_is_bearish = p.close < p.open_price
        p_is_bullish = p.close > p.open_price
        c_is_bullish = c.close > c.open_price
        c_is_bearish = c.close < c.open_price

        # Bullish engulfing: bearish P fully engulfed by bullish C
        is_bullish_engulfing: bool = (
            p_is_bearish
            and c_is_bullish
            and c.open_price < p.close          # C opens below P's close
            and c.close > p.open_price          # C closes above P's open
        )

        # Bearish engulfing: bullish P fully engulfed by bearish C
        is_bearish_engulfing: bool = (
            p_is_bullish
            and c_is_bearish
            and c.open_price > p.close          # C opens above P's close
            and c.close < p.open_price          # C closes below P's open
        )

        # ── BUY signal ────────────────────────────────────────────────────────
        # Condition: P breached lower BB, C re-entered band from below
        if p.close < bb_p.lower and c.close > bb_c.lower:
            if is_bullish_engulfing:
                # MFI above oversold threshold and rising (confirming momentum)
                if mfi_0 > MFI_OVERSOLD and mfi_0 > mfi_1:
                    if c.volume > volume_threshold:
                        signals.append(BuySignal(
                            pair=pair,
                            timestamp=c.hour,
                            price=c.close,
                            rule_id=RULE_ID,
                            confidence=CONFIDENCE,
                        ))

        # ── SELL signal ───────────────────────────────────────────────────────
        # Condition: P breached upper BB, C re-entered band from above
        if p.close > bb_p.upper and c.close < bb_c.upper:
            if is_bearish_engulfing:
                # MFI below overbought threshold and falling (confirming momentum)
                if mfi_0 < MFI_OVERBOUGHT and mfi_0 < mfi_1:
                    if c.volume > volume_threshold:
                        signals.append(SellSignal(
                            pair=pair,
                            timestamp=c.hour,
                            price=c.close,
                            rule_id=RULE_ID,
                            confidence=CONFIDENCE,
                        ))

    return signals