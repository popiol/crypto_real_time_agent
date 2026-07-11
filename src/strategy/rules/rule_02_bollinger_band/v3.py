from __future__ import annotations

import numpy as np
from datetime import datetime
from pydantic import BaseModel, Field

# --- Start of provided data models (for self-contained module) ---
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
    order_book: dict | None = None # Using dict for simplicity, original was OrderBook


class WarmCandle(BaseModel):
    hour: datetime
    open_price: float
    high: float
    low: float
    close: float
    avg_spread_rel: float = 0.0
    volume: float = Field(
        default=0.0,
        description="Average volume_24h of ticks within this hour (proxy for relative market activity)",
    )


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
# --- End of provided data models ---


# Rule constants
BOLLINGER_PERIOD = 20
STD_DEV_MULTIPLIER = 2.0
VOLUME_SMA_PERIOD = 20  # As per pseudocode in the new rule description
VOLUME_STD_DEV_MULTIPLIER = 1.0 # As per pseudocode in the new rule description

# Minimum number of warm candles required to calculate all indicators
MIN_CANDLES_REQUIRED = max(BOLLINGER_PERIOD, VOLUME_SMA_PERIOD)

# Unique identifier for this rule (from idea_id)
RULE_ID = "65c998be-655d-4317-a94e-1c0f989f35ca"


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candle data for calculations
        if len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue

        # Extract close prices and volumes from warm candles
        closes = np.array([c.close for c in pair_data.warm])
        volumes = np.array([c.volume for c in pair_data.warm])

        # --- Calculate Bollinger Bands ---
        # We need at least BOLLINGER_PERIOD candles for this.
        # This check is technically redundant due to MIN_CANDLES_REQUIRED but adds clarity.
        if len(closes) < BOLLINGER_PERIOD:
            continue

        # Calculate Middle Band (SMA of close prices)
        middle_band = np.mean(closes[-BOLLINGER_PERIOD:])

        # Calculate Standard Deviation of close prices
        std_dev = np.std(closes[-BOLLINGER_PERIOD:])

        # Avoid division by zero or nonsensical bands if std_dev is 0
        if std_dev == 0:
            continue

        # Calculate Upper and Lower Bollinger Bands
        upper_band = middle_band + (std_dev * STD_DEV_MULTIPLIER)
        lower_band = middle_band - (std_dev * STD_DEV_MULTIPLIER)

        # --- Calculate Volume SMA and Volume Std Dev ---
        # We need at least VOLUME_SMA_PERIOD candles for this.
        if len(volumes) < VOLUME_SMA_PERIOD:
            continue

        volume_sma = np.mean(volumes[-VOLUME_SMA_PERIOD:])
        volume_std_dev = np.std(volumes[-VOLUME_SMA_PERIOD:])

        # --- Get Current Market Data ---
        # We need at least one hot tick for the current price and timestamp
        if not pair_data.hot:
            continue

        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # Current volume is taken from the latest warm candle, aligning with the historical volume data
        current_volume = pair_data.warm[-1].volume

        # --- Calculate Refined Volume Confirmation Threshold ---
        # This is the core modification: current volume must be above its SMA by at least one standard deviation.
        volume_confirmation_threshold = volume_sma + (volume_std_dev * VOLUME_STD_DEV_MULTIPLIER)

        # --- Generate Signals with Refined Volume Confirmation ---
        # Buy signal: Price closes below Lower Band AND current volume is significantly high
        if current_price < lower_band and current_volume > volume_confirmation_threshold:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=ts,
                    price=current_price,
                    rule_id=RULE_ID,
                )
            )
        # Sell signal: Price closes above Upper Band AND current volume is significantly high
        elif current_price > upper_band and current_volume > volume_confirmation_threshold:
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=ts,
                    price=current_price,
                    rule_id=RULE_ID,
                )
            )

    return signals