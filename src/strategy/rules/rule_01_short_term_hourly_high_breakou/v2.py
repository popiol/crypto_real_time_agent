from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from datetime import datetime

# Rule specific constants
RULE_ID = "hourly-breakout-refined-exit"
CONFIDENCE_LEVEL = 1.0  # Default confidence for a direct rule trigger

# Data sufficiency minimums
MIN_WARM_CANDLES_REQUIRED = 24  # Pseudocode: IF len(data.warm) < 24 THEN RETURN []

# Indicator lookback periods
# Given data.warm has AT MOST 24 entries:
# - RSI(N) and Keltner (which uses EMA/ATR) typically need N+1 points for the first smoothed value.
#   So, for 24 candles, the maximum period N that can produce a final value is 23.
RSI_PERIOD = 23
KELTNER_PERIOD = 23

# - Bollinger Bands (SMA/StdDev) can use N points for an N-period calculation.
#   So, for 24 candles, N can be 24.
BB_PERIOD = 24

# Bollinger Bands constants
BB_K = 2  # Standard deviation multiplier

# Keltner Channels constants
KELTNER_K_ATR = 2  # ATR multiplier

# Entry condition thresholds
RSI_MIN = 60.0
RSI_MAX = 75.0
BB_WIDTH_MIN = 0.05
KELTNER_PCT_MIN = 2.0
KELTNER_PCT_MAX = 6.0
SPREAD_PCT_LIMIT = 1.0


# --- Helper Functions for Indicators ---

def _calculate_sma(data: np.ndarray, period: int) -> float | None:
    """Calculates the Simple Moving Average (SMA) of the last 'period' data points."""
    if len(data) < period:
        return None
    return np.mean(data[-period:])

def _calculate_ema(data: np.ndarray, period: int) -> float | None:
    """
    Calculates the Exponential Moving Average (EMA) for the last data point,
    initialized with an SMA of the first 'period' values.
    """
    if len(data) < period:
        return None
    
    alpha = 2 / (period + 1)
    
    # Initialize EMA with the SMA of the first 'period' values
    ema_current = np.mean(data[:period])

    # Apply EMA formula for subsequent values
    for i in range(period, len(data)):
        ema_current = (data[i] * alpha) + (ema_current * (1 - alpha))
        
    return ema_current

def _calculate_stddev(data: np.ndarray, period: int) -> float | None:
    """Calculates the Standard Deviation of the last 'period' data points."""
    if len(data) < period:
        return None
    return np.std(data[-period:])

def _calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float | None:
    """
    Calculates Average True Range (ATR).
    Uses SMA for smoothing the True Ranges over the 'period'.
    Requires period + 1 candles to calculate 'period' true ranges.
    """
    if len(highs) < period + 1:
        return None

    true_ranges = []
    # Calculate True Ranges for each candle starting from the second
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        true_ranges.append(tr)
    
    # Take SMA of the last 'period' true ranges
    if len(true_ranges) < period: # This check ensures enough TRs are available
        return None
    
    return np.mean(true_ranges[-period:])


def calculate_rsi(closes: list[float], period: int) -> float | None:
    """
    Calculates Relative Strength Index (RSI).
    Requires period + 1 closes to calculate 'period' changes.
    Uses SMA for initial average gain/loss over the period.
    """
    if len(closes) <= period:
        return None

    gains = []
    losses = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))
    
    if len(gains) < period: # Not enough data for initial average
        return None

    # Calculate initial averages using SMA for the last 'period' changes
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0 # Handle division by zero for RS
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands_width(closes: list[float], period: int, k: float) -> float | None:
    """
    Calculates Bollinger Bands Width: (Upper Band - Lower Band) / Middle Band.
    Middle Band is SMA, Upper/Lower bands are k * StdDev from SMA.
    """
    if len(closes) < period:
        return None
    
    closes_arr = np.array(closes)
    
    sma = _calculate_sma(closes_arr, period)
    stddev = _calculate_stddev(closes_arr, period)

    if sma is None or stddev is None or sma == 0:
        return None

    upper_band = sma + (k * stddev)
    lower_band = sma - (k * stddev)
    
    bb_width = (upper_band - lower_band) / sma
    return bb_width

def calculate_keltner_channels_percentage(
    highs: list[float], lows: list[float], closes: list[float], period: int, k_atr: float
) -> float | None:
    """
    Calculates Keltner Channels Percentage as a measure of channel width relative to the middle line:
    (Upper Band - Lower Band) / Middle Line * 100.
    Middle Line is EMA, Bands are k_atr * ATR from Middle Line.
    """
    if len(closes) < period or len(highs) < period or len(lows) < period:
        return None
    
    closes_arr = np.array(closes)
    highs_arr = np.array(highs)
    lows_arr = np.array(lows)

    # Middle Line: EMA of closes
    middle_line = _calculate_ema(closes_arr, period)
    if middle_line is None:
        return None

    # ATR: Average True Range (SMA smoothed for limited data)
    atr = _calculate_atr(highs_arr, lows_arr, closes_arr, period)
    if atr is None:
        return None
    
    upper_band = middle_line + (k_atr * atr)
    lower_band = middle_line - (k_atr * atr)

    if middle_line == 0: # Avoid division by zero
        return None

    keltner_pct = (upper_band - lower_band) / middle_line * 100
    return keltner_pct


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candle data for the lookback (exactly 24 as per pseudocode)
        if len(pair_data.warm) < MIN_WARM_CANDLES_REQUIRED:
            continue

        # Ensure we have at least one hot tick for the current price and spread
        if not pair_data.hot:
            continue

        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        timestamp = current_tick.polled_at

        # Extract data for indicators from the last 24 complete hourly candles
        last_24_candles = pair_data.warm[-MIN_WARM_CANDLES_REQUIRED:]
        
        closes = [c.close for c in last_24_candles]
        highs = [c.high for c in last_24_candles]
        lows = [c.low for c in last_24_candles]

        # Calculate indicators
        # RSI uses RSI_PERIOD due to data constraint (needs N+1 for N changes)
        rsi_val = calculate_rsi(closes, RSI_PERIOD)
        
        # BB uses BB_PERIOD (N points for N-period SMA/StdDev)
        bb_width_val = calculate_bollinger_bands_width(closes, BB_PERIOD, BB_K)
        
        # Keltner uses KELTNER_PERIOD due to data constraint (needs N+1 for N TRs for ATR, and EMA init)
        keltner_pct_val = calculate_keltner_channels_percentage(highs, lows, closes, KELTNER_PERIOD, KELTNER_K_ATR)
        
        # Bid-Ask Spread Percentage
        if current_tick.last_price == 0 or current_tick.bid_price == 0: # Avoid division by zero
            bid_ask_spread_pct = float('inf')
        else:
            bid_ask_spread_pct = (current_tick.ask_price - current_tick.bid_price) / current_tick.last_price * 100

        # Check for None values from indicator calculations (insufficient data for specific indicator period)
        if any(x is None for x in [rsi_val, bb_width_val, keltner_pct_val]):
            continue

        # Determine highest price in the last 24 complete hourly candles for breakout detection
        highest_high_24h = max(c.high for c in last_24_candles)

        # Entry Condition
        # The 'NOT has_open_position()' check from the pseudocode is handled by the trading system
        # which only opens a position if one isn't already active for the pair.
        if (current_price > highest_high_24h and
            RSI_MIN <= rsi_val <= RSI_MAX and
            bb_width_val > BB_WIDTH_MIN and
            KELTNER_PCT_MIN <= keltner_pct_val <= KELTNER_PCT_MAX and
            bid_ask_spread_pct < SPREAD_PCT_LIMIT):

            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=CONFIDENCE_LEVEL,
                )
            )

        # Exit Condition:
        # The pseudocode describes exit logic based on `entry_price` and `highest_price_since_entry`.
        # Per the "NO POSITION VISIBILITY" hard constraint, this rule cannot access such information.
        # Therefore, the fixed stop-loss and trailing stop-loss logic cannot be implemented here.
        # These exit mechanisms are managed by the broader trading system based on its configuration,
        # not by signals from this rule. This rule only generates BuySignals.

    return signals