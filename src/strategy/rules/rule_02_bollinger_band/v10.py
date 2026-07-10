from __future__ import annotations

import numpy as np
from datetime import datetime

# Assuming these imports are available from the context
from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# --- Constants ---
BB_PERIOD = 20  # Bollinger Band lookback period
BB_STD_DEV = 2.0  # Bollinger Band standard deviations
RSI_PERIOD = 14  # RSI lookback period
ATR_PERIOD = 14  # ATR lookback period
VOLATILITY_WINDOW = 60  # Number of past ATR values to consider for mean/std for adaptive thresholds

# Minimum candles needed for all calculations:
# - BB: BB_PERIOD candles
# - RSI: RSI_PERIOD + 1 candles (for `period` deltas)
# - ATR history: To calculate `VOLATILITY_WINDOW` ATR values, each needing `ATR_PERIOD + 1` candles,
#   we need `ATR_PERIOD + VOLATILITY_WINDOW` total candles.
MIN_CANDLES_REQUIRED = max(BB_PERIOD, RSI_PERIOD + 1, ATR_PERIOD + VOLATILITY_WINDOW)

BASE_RSI_OB = 70.0  # Base overbought threshold for RSI
BASE_RSI_OS = 30.0  # Base oversold threshold for RSI
RSI_ADJUSTMENT_FACTOR = 5.0  # Points of RSI threshold adjustment per standard deviation of ATR
MAX_RSI_ADJUSTMENT = 15.0  # Maximum absolute adjustment allowed for RSI thresholds (e.g., 70 +/- 15 = 55 to 85)
MIN_RSI_THRESHOLD_GAP = 5.0  # Minimum points difference between overbought and oversold thresholds

RULE_ID = "bollinger_adaptive_rsi_v1"


# --- Helper Functions ---

def calculate_rsi(closes: np.ndarray, period: int) -> float:
    """
    Calculates the Relative Strength Index (RSI) for the last data point
    using a simplified SMA-based approach for the current RSI value.
    The `closes` array should contain `period + 1` values to calculate `period` price changes.
    """
    if len(closes) < period + 1:
        return np.nan

    # Calculate price changes (deltas will have 'period' elements)
    deltas = np.diff(closes.astype(float))

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # For a single RSI value, we average the gains and losses over the specified period.
    # If `closes` is already a slice of `period + 1` values, then `gains` and `losses`
    # will directly contain `period` elements, so we can just take their mean.
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        # If no losses, RSI is 100. If no gains and no losses, it's neutral (50).
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(candles_segment: list[WarmCandle], period: int) -> float:
    """
    Calculates the Average True Range (ATR) for the last data point in the segment.
    `candles_segment` should contain `period + 1` candles to calculate `period` true ranges.
    """
    if len(candles_segment) < period + 1:
        return np.nan

    true_ranges = []
    # Iterate from the second candle to the last, using the previous candle's close
    for i in range(1, len(candles_segment)):
        high = candles_segment[i].high
        low = candles_segment[i].low
        prev_close = candles_segment[i-1].close
        
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    
    # Average the last 'period' true ranges
    current_atr = np.mean(true_ranges[-period:])
    return current_atr


def get_adaptive_rsi_thresholds(
    current_atr: float,
    atr_history: list[float],  # Historical ATR values used for volatility assessment
    base_ob: float,
    base_os: float,
    adjustment_factor: float,
    max_adjustment: float,
    min_gap: float
) -> tuple[float, float]:
    """
    Dynamically adjusts RSI overbought and oversold thresholds based on market volatility (ATR).
    High volatility widens the thresholds, low volatility tightens them.
    """
    # If not enough history or no variance, use base thresholds
    if len(atr_history) < 2 or np.std(atr_history) == 0:
        return base_ob, base_os

    mean_atr = np.mean(atr_history)
    std_atr = np.std(atr_history)

    # Calculate Z-score of current_atr relative to its recent history
    z_score_atr = (current_atr - mean_atr) / std_atr

    # Adjust thresholds based on z-score, clipped to a maximum adjustment
    adjustment = z_score_atr * adjustment_factor
    adjustment = np.clip(adjustment, -max_adjustment, max_adjustment)

    adaptive_ob = base_ob + adjustment
    adaptive_os = base_os - adjustment

    # Ensure thresholds are within reasonable bounds (e.g., 0-100)
    # and maintain a minimum gap, and OB > OS
    adaptive_ob = np.clip(adaptive_ob, 50 + min_gap, 90.0)
    adaptive_os = np.clip(adaptive_os, 10.0, 50 - min_gap)
    
    # Final check to ensure Overbought threshold is always greater than Oversold threshold
    if adaptive_os >= adaptive_ob:
        # If they cross or are equal, force a separation
        adaptive_os = adaptive_ob - 1.0  # Ensure at least 1 point difference
        # Re-clip to ensure it doesn't go below 10.0
        adaptive_os = np.clip(adaptive_os, 10.0, 90.0 - min_gap)

    return adaptive_ob, adaptive_os


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Bollinger Band with Volatility-Adaptive RSI Confirmation' trading rule.

    A Buy signal is emitted when the price drops below the lower Bollinger Band AND
    the RSI indicates oversold conditions, with the oversold threshold dynamically
    adjusted based on recent market volatility.

    A Sell signal is emitted when the price rises above the upper Bollinger Band AND
    the RSI indicates overbought conditions, with an adaptively adjusted overbought threshold.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for all calculations
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue

        warm_candles = pair_data.warm
        
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # --- 1. Calculate Bollinger Bands ---
        # Get the last BB_PERIOD closing prices
        bb_closes = np.array([c.close for c in warm_candles[-BB_PERIOD:]])
        
        # This check is mostly redundant due to MIN_CANDLES_REQUIRED, but good for clarity
        if len(bb_closes) < BB_PERIOD: 
             continue
        
        bb_mean = np.mean(bb_closes)
        bb_std = np.std(bb_closes)

        if bb_std == 0:  # Avoid division by zero and meaningless bands
            continue

        upper_bb = bb_mean + BB_STD_DEV * bb_std
        lower_bb = bb_mean - BB_STD_DEV * bb_std

        # --- 2. Calculate Relative Strength Index (RSI) ---
        # Get the last RSI_PERIOD + 1 closing prices for RSI calculation
        rsi_closes = np.array([c.close for c in warm_candles[-(RSI_PERIOD + 1):]])
        
        if len(rsi_closes) < RSI_PERIOD + 1:
            continue  # Redundant check, but safe

        current_rsi = calculate_rsi(rsi_closes, RSI_PERIOD)
        if np.isnan(current_rsi):
            continue

        # --- 3. Calculate Volatility (ATR) and its history ---
        # We need `ATR_PERIOD + VOLATILITY_WINDOW` candles to generate `VOLATILITY_WINDOW` ATR values.
        # The slice `warm_candles[-(ATR_PERIOD + VOLATILITY_WINDOW):]` provides this data.
        atr_candles_for_history = warm_candles[-(ATR_PERIOD + VOLATILITY_WINDOW):]
        
        # This check is mostly redundant due to MIN_CANDLES_REQUIRED
        if len(atr_candles_for_history) < ATR_PERIOD + VOLATILITY_WINDOW:
             continue

        atr_values_history = []
        # Iterate to calculate ATR for each window of `ATR_PERIOD + 1` candles
        # This loop correctly generates `VOLATILITY_WINDOW` ATR values, which are then used
        # to assess the historical volatility context.
        for i in range(len(atr_candles_for_history) - ATR_PERIOD):
            sub_candles = atr_candles_for_history[i : i + ATR_PERIOD + 1]
            atr_val = calculate_atr(sub_candles, ATR_PERIOD)
            if not np.isnan(atr_val):
                atr_values_history.append(atr_val)
        
        # Ensure we have enough history to calculate mean/std for adaptive thresholds
        if len(atr_values_history) < 2: 
            continue

        current_atr = atr_values_history[-1]  # The most recent ATR value

        # --- 4. Dynamically adjust RSI thresholds ---
        adaptive_ob_threshold, adaptive_os_threshold = get_adaptive_rsi_thresholds(
            current_atr,
            atr_values_history,  # Pass the entire history for mean/std calculation
            BASE_RSI_OB,
            BASE_RSI_OS,
            RSI_ADJUSTMENT_FACTOR,
            MAX_RSI_ADJUSTMENT,
            MIN_RSI_THRESHOLD_GAP
        )

        # --- 5. Generate Buy/Sell Signals ---
        if current_price < lower_bb and current_rsi < adaptive_os_threshold:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price, rule_id=RULE_ID))
        elif current_price > upper_bb and current_rsi > adaptive_ob_threshold:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price, rule_id=RULE_ID))

    return signals