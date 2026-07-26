from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

# --- Configuration Constants ---
RSI_PERIOD = 14
KCP_PERIOD = 20
BBW_PERIOD = 20
KCP_ATR_MULTIPLIER = 2.0  # Common multiplier for Keltner Channels
BB_STD_MULTIPLIER = 2.0   # Common multiplier for Bollinger Bands
SPREAD_PERCENTAGE_THRESHOLD = 2.0

# Minimum data required for indicators
# RSI(14): requires 14 differences, so 15 close prices (period + 1).
# KCP(20): ATR requires previous close for True Range, so 20 TR values require 21 candles (period + 1).
# BBW(20): requires 20 close prices (period).
# The strictest requirement is 21 warm candles.
MIN_WARM_CANDLES = max(RSI_PERIOD + 1, KCP_PERIOD + 1, BBW_PERIOD)
MIN_HOT_TICKS = 1 # At least one tick for spread_rel

# --- Helper Functions for Indicators ---

def _calculate_sma(data: np.ndarray, period: int) -> float | None:
    """Calculates the Simple Moving Average for the last 'period' elements."""
    if len(data) < period:
        return None
    return np.mean(data[-period:])

def _calculate_std_dev(data: np.ndarray, period: int) -> float | None:
    """Calculates the Standard Deviation for the last 'period' elements."""
    if len(data) < period:
        return None
    # Use ddof=0 for population standard deviation, common in technical indicators
    return np.std(data[-period:], ddof=0)

def _calculate_rsi(closes: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index."""
    if len(closes) < period + 1:
        return None
    
    closes_np = np.array(closes)
    
    # Calculate price changes (differences)
    # np.diff returns an array of length len(closes_np) - 1
    diffs = np.diff(closes_np)
    
    # We need the last 'period' differences to calculate the RSI for the most recent candle.
    relevant_diffs = diffs[-(period):]

    # Separate gains and losses
    gains = relevant_diffs[relevant_diffs > 0]
    losses = relevant_diffs[relevant_diffs < 0]

    # Calculate average gains and losses
    avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
    avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 0.0

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0 # Handle cases with no losses or flat market
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calculate_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float | None:
    """Calculates the Average True Range."""
    # Need period + 1 candles to calculate 'period' True Range values
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None
    
    tr_values = []
    # Iterate for the last 'period' candles to calculate their TR values
    # The loop starts from the index that allows `period` TRs to be calculated,
    # and `closes[i-1]` to be valid.
    for i in range(len(closes) - period, len(closes)):
        current_high = highs[i]
        current_low = lows[i]
        current_close = closes[i]
        previous_close = closes[i-1] # Valid due to `len(closes) >= period + 1` check

        tr = max(current_high - current_low, 
                 abs(current_high - previous_close), 
                 abs(current_low - previous_close))
        tr_values.append(tr)
    
    return _calculate_sma(np.array(tr_values), period) # SMA of the 'period' True Ranges

def _calculate_keltner_channel_percentage(warm_candles: list[WarmCandle], period: int) -> float | None:
    """Calculates the Keltner Channel Percentage (KCP)."""
    # Requires period + 1 candles for ATR calculation
    if len(warm_candles) < period + 1:
        return None

    closes = [c.close for c in warm_candles]
    highs = [c.high for c in warm_candles]
    lows = [c.low for c in warm_candles]

    # Middle Band (using SMA for simplicity and short lookback, instead of EMA)
    mb = _calculate_sma(np.array(closes), period)
    if mb is None:
        return None

    # ATR
    atr = _calculate_atr(highs, lows, closes, period)
    if atr is None:
        return None

    # Upper and Lower Bands
    ub = mb + KCP_ATR_MULTIPLIER * atr
    lb = mb - KCP_ATR_MULTIPLIER * atr

    # KCP calculation for the most recent close price
    current_close = closes[-1]
    
    if (ub - lb) == 0: # Avoid division by zero, implies no channel width
        return 0.0 
    
    kcp = ((current_close - lb) / (ub - lb)) * 100
    return kcp

def _calculate_bollinger_bands_width(warm_candles: list[WarmCandle], period: int) -> float | None:
    """Calculates the Bollinger Bands Width (BBW)."""
    if len(warm_candles) < period:
        return None

    closes_np = np.array([c.close for c in warm_candles])
    
    # Calculate SMA and STD DEV over the last 'period' closes
    relevant_closes = closes_np[-period:]

    mb = _calculate_sma(relevant_closes, period)
    std_dev = _calculate_std_dev(relevant_closes, period)

    if mb is None or std_dev is None:
        return None

    # Upper and Lower Bands
    ub = mb + BB_STD_MULTIPLIER * std_dev
    lb = mb - BB_STD_MULTIPLIER * std_dev

    # BBW calculation
    if mb == 0: # Avoid division by zero, implies zero price
        return 0.0
    
    bbw = ((ub - lb) / mb) * 100
    return bbw

def _calculate_bid_ask_spread_percentage(hot_ticks: list[Tick]) -> float | None:
    """Retrieves the relative bid-ask spread from the most recent tick."""
    if not hot_ticks:
        return None
    return hot_ticks[-1].spread_rel

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Moderate Uptrend Momentum Entry with Liquidity Filter' rule.
    Triggers a BUY signal when RSI (65-75), KCP (140-175), BBW (5-15) are within
    specified ranges, and bid-ask spread is low (< 2.0%).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # 1. Check for sufficient data for all indicators
        if len(warm_candles) < MIN_WARM_CANDLES:
            continue
        if len(hot_ticks) < MIN_HOT_TICKS:
            continue

        # 2. Calculate all required indicators
        rsi_val = _calculate_rsi([c.close for c in warm_candles], RSI_PERIOD)
        kcp_val = _calculate_keltner_channel_percentage(warm_candles, KCP_PERIOD)
        bbw_val = _calculate_bollinger_bands_width(warm_candles, BBW_PERIOD)
        spread_val = _calculate_bid_ask_spread_percentage(hot_ticks)

        # Ensure all indicators were calculated successfully (i.e., not None)
        if any(v is None for v in [rsi_val, kcp_val, bbw_val, spread_val]):
            continue

        # 3. Apply the rule conditions
        rsi_condition = (rsi_val >= 65) and (rsi_val <= 75)
        kcp_condition = (kcp_val >= 140) and (kcp_val <= 175)
        bbw_condition = (bbw_val >= 5) and (bbw_val <= 15)
        spread_condition = (spread_val < SPREAD_PERCENTAGE_THRESHOLD)

        # 4. If all conditions are met, generate a BuySignal
        if rsi_condition and kcp_condition and bbw_condition and spread_condition:
            signals.append(BuySignal(
                pair=pair,
                timestamp=hot_ticks[-1].polled_at,
                price=hot_ticks[-1].last_price,
                rule_id="momentum_uptrend_entry_v1",
                confidence=None, # Confidence is not specified in the rule
                indicators={
                    "rsi": rsi_val,
                    "kcp": kcp_val,
                    "bbw": bbw_val,
                    "spread_rel": spread_val,
                }
            ))
            
    return signals