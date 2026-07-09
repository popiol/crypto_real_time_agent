from __future__ import annotations
import numpy as np
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
import math

# Re-define data models to make the module self-contained,
# as per the instruction "Return ONLY the raw Python source code".
class OrderBook(BaseModel):
    bids: List[List[float]]
    asks: List[List[float]]

class Tick(BaseModel):
    pair: str
    polled_at: datetime
    last_price: float
    bid_price: float
    bid_volume: float
    ask_price: float
    ask_volume: float
    volume_24h: float = 0.0
    mid_price: float
    spread_abs: float
    spread_rel: float
    order_book: Optional[OrderBook] = None

class WarmCandle(BaseModel):
    hour: datetime
    open_price: float
    high: float
    low: float
    close: float
    avg_spread_rel: float = 0.0

class ColdMonth(BaseModel):
    month: str
    min_price: float
    max_price: float
    avg_price: float
    avg_daily_spread: float
    candle_count: int
    last_candle_hour: datetime

class PairData(BaseModel):
    hot: List[Tick] = Field(default=[], description="TTL-capped; ~300 ticks at 1 poll/sec with default 300s retention")
    warm: List[WarmCandle] = Field(default=[], description="At most 24 entries (last 24 hourly candles)")
    cold: List[ColdMonth] = Field(default=[], description="One entry per calendar month; unbounded")

class BuySignal(BaseModel):
    pair: str
    timestamp: datetime
    price: float
    rule_id: str = ""
    confidence: Optional[float] = None

class SellSignal(BaseModel):
    pair: str
    timestamp: datetime
    price: float
    rule_id: str = ""
    confidence: Optional[float] = None

MarketData = Dict[str, PairData]

# Rule Parameters
RSI_PERIOD = 14
SHORT_MA_PERIOD = 20
LONG_MA_PERIOD = 50
DIVERGENCE_LOOKBACK_PERIOD = 60 # Number of candles to look back for divergence patterns

# --- Helper Functions for Indicators ---

def calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates the Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return np.full(len(prices), np.nan)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    rsi = np.zeros_like(prices)

    # Initial average gain/loss (simple average over first 'period' deltas)
    # Note: deltas array is len(prices) - 1, so gains/losses are also len(prices) - 1
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # Smoothed average for subsequent periods
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    # Calculate RS and RSI
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates the Simple Moving Average (SMA)."""
    if len(prices) < period:
        return np.full(len(prices), np.nan)
    
    sma = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1 : i + 1])
    return sma

# --- Divergence Detection Helpers ---

def find_pivot_low(prices: np.ndarray, rsi_values: np.ndarray, current_idx: int, lookback_period: int) -> Optional[int]:
    """
    Finds a pivot low within the lookback period for bullish divergence.
    A pivot low is defined as a candle whose close is lower than its immediate neighbors.
    """
    # A pivot candidate `i` needs `prices[i-1]` and `prices[i+1]` to exist.
    # So `i` must be at least 1 and at most `len(prices) - 2`.
    # The search range is from `current_idx - lookback_period` to `current_idx - 1`.
    # Combining these, the actual search range for `i` is:
    start_search_idx = max(1, current_idx - lookback_period)
    end_search_idx = current_idx - 2 # `i` cannot be `current_idx - 1` because `i+1` would be `current_idx` (current candle)

    if end_search_idx < start_search_idx: # Not enough range to find a valid pivot
        return None

    for i in range(end_search_idx, start_search_idx - 1, -1):
        # Check if it's a local low in price (lower than immediate neighbors)
        is_price_local_low = (prices[i] < prices[i-1] and prices[i] < prices[i+1])
        # Check if it's a local low in RSI
        is_rsi_local_low = (rsi_values[i] < rsi_values[i-1] and rsi_values[i] < rsi_values[i+1])

        if is_price_local_low and is_rsi_local_low:
            # Bullish divergence conditions:
            # 1. Current price is a lower low than the pivot price (`price[i] > CURRENT_PRICE`)
            # 2. Current RSI is a higher low than the pivot RSI (`RSI[i] < CURRENT_RSI`)
            if prices[current_idx] < prices[i] and rsi_values[current_idx] > rsi_values[i]:
                 return i # Return the index of the most recent valid pivot
    return None

def find_pivot_high(prices: np.ndarray, rsi_values: np.ndarray, current_idx: int, lookback_period: int) -> Optional[int]:
    """
    Finds a pivot high within the lookback period for bearish divergence.
    A pivot high is defined as a candle whose close is higher than its immediate neighbors.
    """
    start_search_idx = max(1, current_idx - lookback_period)
    end_search_idx = current_idx - 2

    if end_search_idx < start_search_idx:
        return None

    for i in range(end_search_idx, start_search_idx - 1, -1):
        # Check if it's a local high in price
        is_price_local_high = (prices[i] > prices[i-1] and prices[i] > prices[i+1])
        # Check if it's a local high in RSI
        is_rsi_local_high = (rsi_values[i] > rsi_values[i-1] and rsi_values[i] > rsi_values[i+1])

        if is_price_local_high and is_rsi_local_high:
            # Bearish divergence conditions:
            # 1. Current price is a higher high than the pivot price (`price[i] < CURRENT_PRICE`)
            # 2. Current RSI is a lower high than the pivot RSI (`RSI[i] > CURRENT_RSI`)
            if prices[current_idx] > prices[i] and rsi_values[current_idx] < rsi_values[i]:
                return i # Return the index of the most recent valid pivot
    return None


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    
    # Calculate minimum required candles for all indicators and lookbacks.
    # For RSI and MAs to have a valid value at `current_idx` (last candle),
    # we need at least `max(RSI_PERIOD, SHORT_MA_PERIOD, LONG_MA_PERIOD)` candles.
    # For divergence detection, we look back `DIVERGENCE_LOOKBACK_PERIOD` candles from `current_idx`.
    # A pivot candidate `i` must be at least `current_idx - DIVERGENCE_LOOKBACK_PERIOD`.
    # Also, a pivot `i` needs `i-1` and `i+1` to exist for the "local low/high" check.
    # This means `i` must be between `1` and `len(prices) - 2`.
    # The latest possible pivot `i` is `current_idx - 2`.
    # So, the number of candles needed is `max(max_indicator_period, DIVERGENCE_LOOKBACK_PERIOD + 1)`.
    # `max_indicator_period = max(RSI_PERIOD, SHORT_MA_PERIOD, LONG_MA_PERIOD) = max(14, 20, 50) = 50`.
    # `DIVERGENCE_LOOKBACK_PERIOD + 1 = 60 + 1 = 61`.
    # Thus, `MIN_REQUIRED_CANDLES = max(50, 61) = 61`.
    # The `warm` data list contains "At most 24 entries".
    # This rule, as specified by its parameters, will almost certainly never generate a signal
    # due to insufficient historical data from the `warm` candle list.
    MIN_REQUIRED_CANDLES = max(RSI_PERIOD, SHORT_MA_PERIOD, LONG_MA_PERIOD, DIVERGENCE_LOOKBACK_PERIOD + 1)

    for pair, pair_data in data.items():
        candles = pair_data.warm
        
        if len(candles) < MIN_REQUIRED_CANDLES:
            # Insufficient data to calculate all indicators and lookbacks.
            # print(f"Insufficient warm candle data for {pair}. Required: {MIN_REQUIRED_CANDLES}, Available: {len(candles)}")
            continue 
        
        close_prices = np.array([c.close for c in candles])
        
        # Calculate indicators
        rsi_values = calculate_rsi(close_prices, RSI_PERIOD)
        short_ma = calculate_sma(close_prices, SHORT_MA_PERIOD)
        long_ma = calculate_sma(close_prices, LONG_MA_PERIOD)
        
        current_idx = len(close_prices) - 1
        
        # Ensure the latest indicator values are valid (not NaN).
        # This check is crucial if `MIN_REQUIRED_CANDLES` is just at the threshold.
        if np.isnan(rsi_values[current_idx]) or \
           np.isnan(short_ma[current_idx]) or \
           np.isnan(long_ma[current_idx]):
            continue
            
        current_price = close_prices[current_idx]
        current_rsi = rsi_values[current_idx]
        current_short_ma = short_ma[current_idx]
        current_long_ma = long_ma[current_idx]
        
        # Previous MA values for crossover detection.
        # Need at least 2 candles for previous MA values.
        if current_idx < 1 or np.isnan(short_ma[current_idx - 1]) or np.isnan(long_ma[current_idx - 1]):
            continue
            
        prev_short_ma = short_ma[current_idx - 1]
        prev_long_ma = long_ma[current_idx - 1]

        # --- Divergence Detection ---
        bullish_divergence = False
        bearish_divergence = False

        pivot_low_idx = find_pivot_low(close_prices, rsi_values, current_idx, DIVERGENCE_LOOKBACK_PERIOD)
        if pivot_low_idx is not None:
            bullish_divergence = True

        pivot_high_idx = find_pivot_high(close_prices, rsi_values, current_idx, DIVERGENCE_LOOKBACK_PERIOD)
        if pivot_high_idx is not None:
            bearish_divergence = True

        # --- MA Crossover Confirmation ---
        bullish_ma_crossover = (current_short_ma > current_long_ma and prev_short_ma <= prev_long_ma)
        bearish_ma_crossover = (current_short_ma < current_long_ma and prev_short_ma >= prev_long_ma)

        # --- Signal Generation ---
        if bullish_divergence and bullish_ma_crossover:
            signals.append(BuySignal(
                pair=pair,
                timestamp=candles[current_idx].hour,
                price=current_price,
                rule_id="db1780d8-07d8-4f56-8d48-387f1db3c71f"
            ))
        elif bearish_divergence and bearish_ma_crossover:
            signals.append(SellSignal(
                pair=pair,
                timestamp=candles[current_idx].hour,
                price=current_price,
                rule_id="db1780d8-07d8-4f56-8d48-387f1db3c71f"
            ))
            
    return signals