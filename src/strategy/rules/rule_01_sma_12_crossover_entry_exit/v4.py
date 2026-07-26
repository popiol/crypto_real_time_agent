from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

# Rule ID
RULE_ID = "oversold_mean_reversion_v1"

# Constants for indicator periods and thresholds
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STDDEV = 2  # Standard deviation multiplier for Bollinger Bands
SMA_PERIOD = 12

# Buy signal thresholds (Oversold conditions)
RSI_BUY_THRESHOLD = 30
BB_PERCENT_B_BUY_THRESHOLD = 0.2

# Sell signal (exit long) thresholds
RSI_SELL_THRESHOLD = 50  # RSI recovery

# Minimum data requirements for indicators
# RSI_PERIOD = 14 requires 14+1=15 candles for the first RSI value.
# BB_PERIOD = 20 requires 20 candles.
# SMA_PERIOD = 12 requires 12 candles.
MIN_WARM_CANDLES_REQUIRED = max(RSI_PERIOD + 1, BB_PERIOD, SMA_PERIOD)
MIN_HOT_TICKS = 1  # Only the most recent tick is needed for current_price

# Helper functions for indicator calculations
def calculate_sma(closes: list[float], period: int) -> float | None:
    """Calculates the Simple Moving Average."""
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))

def calculate_rsi(closes: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI)."""
    if len(closes) < period + 1:
        return None

    price_array = np.array(closes, dtype=float)
    deltas = np.diff(price_array)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(gains)
    avg_loss = np.zeros_like(losses)

    # Initial average gain/loss over the first 'period' deltas
    # Requires 'period' deltas, which means 'period + 1' closes.
    if len(gains) < period:
        return None

    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    # Smoothed average gain/loss for subsequent periods
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

    last_avg_gain = avg_gain[-1]
    last_avg_loss = avg_loss[-1]

    if last_avg_loss == 0:
        return 100.0 if last_avg_gain > 0 else 0.0
    
    rs = last_avg_gain / last_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(closes: list[float], period: int, stddev_multiplier: float) -> tuple[float | None, float | None, float | None]:
    """Calculates Bollinger Bands (Upper, Middle, Lower)."""
    if len(closes) < period:
        return None, None, None

    closes_arr = np.array(closes[-period:], dtype=float)
    
    middle_band = float(np.mean(closes_arr))
    std_dev = float(np.std(closes_arr))

    upper_band = middle_band + (std_dev * stddev_multiplier)
    lower_band = middle_band - (std_dev * stddev_multiplier)

    return upper_band, middle_band, lower_band

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Oversold Mean Reversion Entry with Dynamic Exits rule.
    
    Initiates a BuySignal when Hourly RSI(14) < 30, Hourly Bollinger Band %B(20) < 0.2,
    and the current price is below the Hourly 12-period Simple Moving Average.
    
    Emits a SellSignal (to close an existing long position) when Hourly RSI(14) recovers above 50.
    
    Note: Profit target (2%) and stop loss (1%) exit conditions, as described in the pseudocode,
    are dependent on the position's entry price. Since 'entry_price' is not available
    within the 'MarketData' object passed to this rule's 'signal' function, these specific
    exit conditions cannot be implemented here. They are expected to be handled by the
    trading agent that consumes these signals and manages open positions.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm data for all indicators
        if len(pair_data.warm) < MIN_WARM_CANDLES_REQUIRED:
            continue

        # Ensure enough hot data for current price
        if not pair_data.hot:
            continue

        # Extract relevant data
        hourly_closes = [candle.close for candle in pair_data.warm]
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_tick_timestamp = current_tick.polled_at

        # --- Calculate Indicators ---
        # SMA(12)
        sma_12 = calculate_sma(hourly_closes, SMA_PERIOD)
        if sma_12 is None: continue

        # RSI(14) - based on hourly closes
        current_rsi = calculate_rsi(hourly_closes, RSI_PERIOD)
        if current_rsi is None: continue

        # Bollinger Bands (20, 2) and %B - BB are based on hourly closes, %B uses current_price
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(hourly_closes, BB_PERIOD, BB_STDDEV)
        
        # Handle cases where BB cannot be calculated or bands are collapsed (std_dev = 0)
        if bb_upper is None or bb_lower is None or (bb_upper - bb_lower) == 0: 
            # If bands are undefined or collapsed, %B is not meaningful for this rule.
            # Skip signal generation for this pair in this case.
            continue
        
        percent_b = (current_price - bb_lower) / (bb_upper - bb_lower)

        # Store indicators for potential signal
        indicators = {
            "sma_12": sma_12,
            "current_rsi": current_rsi,
            "percent_b": percent_b,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
        }

        # --- Entry Condition (Long Only) ---
        # A BuySignal is emitted if oversold conditions are met.
        # The agent will typically check if there's an open position before acting on this.
        if (current_rsi < RSI_BUY_THRESHOLD and
            percent_b < BB_PERCENT_B_BUY_THRESHOLD and
            current_price < sma_12):
            
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=current_tick_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    indicators=indicators
                )
            )
        
        # --- Exit Condition (Close Long Position) ---
        # A SellSignal is emitted if the RSI recovers.
        # This signal is intended to close an existing long position.
        # The agent consuming this signal will decide whether a position exists.
        elif current_rsi > RSI_SELL_THRESHOLD:
            
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=current_tick_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    exit_reason="RSI_Recovery_Exit",
                    indicators=indicators
                )
            )

    return signals