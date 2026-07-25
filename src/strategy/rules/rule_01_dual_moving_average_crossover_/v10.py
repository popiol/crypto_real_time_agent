from __future__ import annotations
import numpy as np
from datetime import datetime
from pydantic import BaseModel, Field

# --- Data Models (Copied from the provided context) ---
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
    order_book: dict | None = None # Using dict for simplicity, actual OrderBook class not provided

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
# --- End of Data Models ---


# Constants for Bollinger Bands and RSI
BB_PERIOD = 20
BB_STD_DEV = 2.0
RSI_PERIOD = 14

# Relaxed thresholds for 'bb-rsi-relax-001'
BB_PERCENT_B_LONG_THRESHOLD = 0.5
RSI_LONG_THRESHOLD = 30
BB_PERCENT_B_SHORT_THRESHOLD = 0.5
RSI_SHORT_THRESHOLD = 70

RULE_ID = "bb-rsi-relax-001"


def _calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates Simple Moving Average (SMA) for a given numpy array of prices.
    """
    if len(prices) < period:
        return np.array([])
    # Using convolution for efficiency
    sma = np.convolve(prices, np.ones(period), 'valid') / period
    return sma

def _calculate_bollinger_bands(prices: np.ndarray, period: int, num_std_dev: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates Bollinger Bands (Middle, Upper, Lower) for a given numpy array of prices.
    Returns (middle_band, upper_band, lower_band).
    """
    if len(prices) < period:
        return np.array([]), np.array([]), np.array([])

    middle_band = _calculate_sma(prices, period)

    # Calculate standard deviation for each window
    std_devs = np.array([
        np.std(prices[i - period + 1 : i + 1])
        for i in range(period - 1, len(prices))
    ])

    upper_band = middle_band + (std_devs * num_std_dev)
    lower_band = middle_band - (std_devs * num_std_dev)

    return middle_band, upper_band, lower_band

def _calculate_bollinger_percent_b(prices: np.ndarray, period: int, num_std_dev: float) -> np.ndarray:
    """
    Calculates Bollinger Bands %B for a given numpy array of prices.
    %B = (Current Close - Lower Band) / (Upper Band - Lower Band)
    """
    _, upper_band, lower_band = _calculate_bollinger_bands(prices, period, num_std_dev)

    if len(upper_band) == 0:
        return np.array([])

    # The prices array used for %B calculation should align with the bands.
    # The bands start after `period - 1` prices have passed.
    # So, we need to take `prices[period-1:]` for the numerator calculation.
    relevant_prices = prices[period - 1:]

    denominator = upper_band - lower_band
    # Handle division by zero: if bands are identical, %B is undefined or 0.5 (middle).
    # Replace 0 denominators with NaN to avoid RuntimeWarning and allow filtering later.
    denominator = np.where(denominator == 0, np.nan, denominator)

    percent_b = (relevant_prices - lower_band) / denominator
    return percent_b

def _calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates Relative Strength Index (RSI) for a given numpy array of prices.
    """
    if len(prices) < period + 1: # Need at least period + 1 prices to get period differences
        return np.array([])

    diffs = np.diff(prices)
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0) # Make losses positive

    rsi_values = np.full(len(prices) - period, np.nan, dtype=float)

    # Initial average gain and loss for the first `period` differences
    # Only consider the first `period` differences for the initial average calculation
    initial_gains = gains[:period]
    initial_losses = losses[:period]

    avg_gain = np.mean(initial_gains)
    avg_loss = np.mean(initial_losses)

    if avg_loss == 0: # Avoid division by zero
        rs = np.inf if avg_gain > 0 else 0.0 # If no losses but gains, RS is inf. If no changes, RS is 0.
    else:
        rs = avg_gain / avg_loss

    rsi_values[0] = 100 - (100 / (1 + rs))

    # Wilder's smoothing for subsequent values
    for i in range(period, len(diffs)):
        current_gain = gains[i]
        current_loss = losses[i]

        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period

        if avg_loss == 0:
            rs = np.inf if avg_gain > 0 else 0.0
        else:
            rs = avg_gain / avg_loss

        rsi_values[i - period + 1] = 100 - (100 / (1 + rs))

    return rsi_values

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Relaxed Bollinger Bands %B Thresholds for Mean Reversion ('bb-rsi-relax-001').

    A buy signal is generated when Bollinger Bands %B falls below 0.5
    AND the 14-period RSI is simultaneously below 30.

    A sell signal is generated when Bollinger Bands %B rises above 0.5
    AND the 14-period RSI is simultaneously above 70.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_data = pair_data.warm

        # We need enough data for both BB (20) and RSI (14).
        # BB needs `BB_PERIOD` candles for the first %B value.
        # RSI needs `RSI_PERIOD + 1` candles for the first RSI value.
        # So, we need at least max(BB_PERIOD, RSI_PERIOD + 1) candles.
        required_data_points = max(BB_PERIOD, RSI_PERIOD + 1)
        if len(warm_data) < required_data_points:
            continue

        # Extract 'close' prices from each WarmCandle object.
        # These are assumed to be chronologically ordered, with the latest price at the end.
        prices = np.array([candle.close for candle in warm_data], dtype=float)

        # Calculate indicators
        bb_percent_b_series = _calculate_bollinger_percent_b(prices, BB_PERIOD, BB_STD_DEV)
        rsi_series = _calculate_rsi(prices, RSI_PERIOD)

        # Ensure we have at least one valid value for both indicators
        if len(bb_percent_b_series) == 0 or len(rsi_series) == 0:
            continue

        # The last value of each series corresponds to the latest available price.
        last_bb_percent_b = bb_percent_b_series[-1]
        last_rsi = rsi_series[-1]

        # Determine the most recent price and timestamp for the signal.
        # Prioritize 'hot' (tick) data for the most up-to-date information,
        # otherwise fall back to the latest 'warm' (hourly candle) data.
        current_price = None
        timestamp = None

        if pair_data.hot:
            current_price = pair_data.hot[-1].last_price
            timestamp = pair_data.hot[-1].polled_at
        elif pair_data.warm:
            # Use the close price of the very last warm candle
            current_price = pair_data.warm[-1].close
            timestamp = pair_data.warm[-1].hour
        else:
            # No current price available from hot or warm data. Cannot generate a relevant signal.
            continue

        # Check for potential NaN/Inf in indicator values (e.g., from division by zero or flat price series)
        if not np.isfinite(last_bb_percent_b) or not np.isfinite(last_rsi):
            continue

        # Determine trading signal based on the NEW relaxed rule logic
        if last_bb_percent_b < BB_PERCENT_B_LONG_THRESHOLD and last_rsi < RSI_LONG_THRESHOLD:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID
            ))
        elif last_bb_percent_b > BB_PERCENT_B_SHORT_THRESHOLD and last_rsi > RSI_SHORT_THRESHOLD:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID
            ))

    return signals