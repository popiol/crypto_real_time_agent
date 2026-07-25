from __future__ import annotations
import numpy as np
from datetime import datetime
from pydantic import BaseModel, Field

# --- Data Models (provided in the problem description) ---
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


# Constants for indicators
RSI_PERIOD = 14
BB_PERIOD = 20
BB_K = 2.0 # Standard deviation multiplier for Bollinger Bands

# Rule-specific constants based on the idea "RSI_BB_Relaxed_Thresholds_V1"
RULE_ID = "RSI_BB_Relaxed_Thresholds_V1"
RSI_BUY_THRESHOLD = 35              # Relaxed from 30 to 35
RSI_SELL_THRESHOLD = 65             # Relaxed from 70 to 65
BB_PERCENT_B_BUY_THRESHOLD = 0.2    # Unchanged, as per idea's description
BB_PERCENT_B_SELL_THRESHOLD = 0.8   # Unchanged, as per idea's description


def _calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates Relative Strength Index (RSI) for a given numpy array of prices.
    Returns an array of RSI values, with the latest RSI at the end.
    """
    if len(prices) < period + 1: # Need at least period + 1 prices to get period differences
        return np.array([])

    diffs = np.diff(prices)
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0) # Make losses positive

    rsi_values = np.full(len(prices) - period, np.nan, dtype=float)

    # Initial average gain and loss for the first `period` differences
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


def _calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates Simple Moving Average (SMA) for a given numpy array of prices.
    The result array will have length len(prices) - period + 1.
    """
    if len(prices) < period:
        return np.array([])
    # Using np.convolve for efficiency
    return np.convolve(prices, np.ones(period), 'valid') / period


def _calculate_bollinger_bands_percent_b(prices: np.ndarray, period: int, k: float) -> np.ndarray:
    """
    Calculates Bollinger Bands %B for a given numpy array of prices.
    %B = (Current Close - Lower Band) / (Upper Band - Lower Band)
    """
    if len(prices) < period:
        return np.array([])

    sma_series = _calculate_sma(prices, period)
    
    # Calculate rolling standard deviation over the same `period` window as SMA
    std_dev_series = np.array([np.std(prices[i:i+period]) for i in range(len(prices) - period + 1)])

    upper_band = sma_series + (std_dev_series * k)
    lower_band = sma_series - (std_dev_series * k)

    # The 'current closes' for %B calculation are the prices that correspond to each SMA/band point.
    # These are the prices from `period-1` index to the end of the `prices` array.
    current_closes_for_bands = prices[period - 1:]

    # Avoid division by zero if upper_band == lower_band (e.g., flat prices)
    denominator = upper_band - lower_band
    # If denominator is zero, %B is typically undefined or taken as 0.5 (mid-band).
    percent_b = np.where(denominator != 0, (current_closes_for_bands - lower_band) / denominator, 0.5)

    return percent_b


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'RSI_BB_Relaxed_Thresholds_V1' trading rule.

    This rule generates a buy signal when the 14-period Relative Strength Index (RSI) is below 35
    AND the 20-period, 2-standard-deviation Bollinger Bands %B is below 0.2.
    A sell signal is generated when RSI is above 65 AND Bollinger Bands %B is above 0.8.
    This aims to increase signal generation by relaxing RSI thresholds while maintaining
    Bollinger Bands %B conditions for mean reversion opportunities.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_data = pair_data.warm

        # Determine the maximum number of candles required for all indicators
        # RSI needs period + 1. BB needs period.
        required_data_points = max(RSI_PERIOD + 1, BB_PERIOD)

        if len(warm_data) < required_data_points:
            # Not enough data to calculate indicators reliably
            continue

        # Extract 'close' prices from each WarmCandle object.
        # These are assumed to be chronologically ordered, with the latest price at the end.
        prices = np.array([candle.close for candle in warm_data], dtype=float)

        # Calculate indicators
        rsi_series = _calculate_rsi(prices, RSI_PERIOD)
        bb_percent_b_series = _calculate_bollinger_bands_percent_b(prices, BB_PERIOD, BB_K)

        # Ensure we have at least one valid value for each indicator
        # The length of RSI series is len(prices) - period, and BB%B is len(prices) - period + 1.
        # We need the latest value from both. The required_data_points check already ensures this.
        if len(rsi_series) == 0 or len(bb_percent_b_series) == 0:
            continue

        # Get the latest values for each indicator
        last_rsi = rsi_series[-1]
        last_bb_percent_b = bb_percent_b_series[-1]

        # Check for potential NaN/Inf in indicator values before using them
        if not (np.isfinite(last_rsi) and np.isfinite(last_bb_percent_b)):
            continue

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
        
        # Apply the rule's conditions for buy signal: RSI < 35 AND BB%B < 0.2
        if (last_rsi < RSI_BUY_THRESHOLD) and (last_bb_percent_b < BB_PERCENT_B_BUY_THRESHOLD):
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID
            ))
        # Apply the rule's conditions for sell signal: RSI > 65 AND BB%B > 0.8
        elif (last_rsi > RSI_SELL_THRESHOLD) and (last_bb_percent_b > BB_PERCENT_B_SELL_THRESHOLD):
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID
            ))

    return signals