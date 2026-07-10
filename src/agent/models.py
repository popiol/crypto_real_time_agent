from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class OrderBookLevel(BaseModel):
    price: float
    volume: float
    timestamp: int


class OrderBook(BaseModel):
    asks: list[OrderBookLevel]
    bids: list[OrderBookLevel]


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
    order_book: OrderBook | None = None


class WarmCandle(BaseModel):
    hour: datetime
    open_price: float
    high: float
    low: float
    close: float
    avg_spread_rel: float = 0.0
    volume: float = Field(default=0.0, description="Average volume_24h of ticks within this hour (proxy for relative market activity)")


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


MarketData = dict[str, PairData]


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


class AppConfig(BaseModel):
    pairs: list[str] | None = None  # None → auto-discover all *USD pairs from Kraken
    data_dir: str = "data"
    state_dir: str = "data/state"
    backtest_data_dir: str = "../crypto_alerts_llm/data/raw"
    hot_tier_retention_seconds: int = 300
    min_poll_interval_seconds: float = 1.0
    backoff_initial_seconds: float = 2.0
    backoff_max_seconds: float = 60.0
    llm_model: str = "gemini-2.0-flash"
    rule_min_signals: int = 20
    rule_mature_days: int = 7  # evaluation days before "mature" logic applies
    rule_early_deprecation_gain: float = (
        -0.10
    )  # deprecate immediately below this avg gain (ratio)
    rule_mature_deprecation_gain: float = (
        0.0  # deprecate mature rule at or below this avg gain
    )
    rule_zero_signal_max_cycles: int = 7  # deprecate a rule that emits 0 signals for this many consecutive analysis cycles
