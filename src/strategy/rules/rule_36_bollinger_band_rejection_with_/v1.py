from __future__ import annotations
import numpy as np
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Parameters ---
BB_PERIOD = 20
BB_DEV = 2.0
MFI_PERIOD = 14
MFI_OVERSOLD = 20
MFI_OVERBOUGHT = 80
VOLUME_MA_PERIOD = 20
BBW_MA_PERIOD = 10
PIN_BAR_WICK_RATIO = 0.6  # Minimum ratio of wick to total candle range
PIN_BAR_BODY_RATIO = 0.3  # Maximum ratio of body to total candle range

# Minimum number of candles required for all indicators to be calculated and aligned.
# BB: BB_PERIOD candles
# MFI: MFI_PERIOD + 3 candles (for MFI[2] for reversal check)
# Volume MA: VOLUME_MA_PERIOD candles
# BBW MA: BB_PERIOD candles to get BB_WIDTH, then BBW_MA_PERIOD candles of BB_WIDTH.
#         So, len(closes) - BB_PERIOD + 1 >= BBW_MA_PERIOD
#         len(closes) >= BB_PERIOD + BBW_MA_PERIOD - 1
MIN_CANDLES_REQUIRED = max(
    BB_PERIOD,
    MFI_PERIOD + 3,
    VOLUME_MA_PERIOD,
    BB_PERIOD + BBW_MA_PERIOD - 1
)

def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < period:
        return np.array([])
    return np.convolve(data, np.ones(period)/period, mode='valid')

def _calculate_stddev(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Standard Deviation."""
    if len(data) < period:
        return np.array([])
    # Use rolling window for standard deviation
    stddevs = [np.std(data[i-period+1 : i+1]) for i in range(period-1, len(data))]
    return np.array(stddevs)

def _calculate_mfi(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray, period: int) -> np.ndarray:
    """Calculates Money Flow Index (MFI)."""
    if len(highs) < period + 1: # Need period + 1 for initial TP comparison
        return np.array([])

    typical_prices = (highs + lows + closes) / 3
    money_flow = typical_prices * volumes

    mfi_values = np.full(len(highs), np.nan)

    for i in range(period, len(highs)):
        positive_money_flow = 0.0
        negative_money_flow = 0.0

        # Sum money flow over the period
        for j in range(i - period + 1, i + 1):
            if typical_prices[j] > typical_prices[j-1]:
                positive_money_flow += money_flow[j]
            elif typical_prices[j] < typical_prices[j-1]:
                negative_money_flow += money_flow[j]

        if negative_money_flow == 0:
            # Handle division by zero: if no negative money flow, MFI is 100 (if positive flow exists), else 50 (if no flow at all)
            mfi_values[i] = 100 if positive_money_flow > 0 else 50
        else:
            money_flow_ratio = positive_money_flow / negative_money_flow
            mfi_values[i] = 100 - (100 / (1 + money_flow_ratio))
            
    # Return only valid MFI values, starting from the first calculated point
    return mfi_values[period:]

def _is_bullish_pin_bar(open_p: float, high_p: float, low_p: float, close_p: float) -> bool:
    """Detects a bullish pin bar based on pseudocode ratios."""
    total_range = high_p - low_p
    if total_range == 0:
        return False

    if close_p > open_p: # Green candle
        lower_wick = open_p - low_p
        upper_wick = high_p - close_p
        body = close_p - open_p
    else: # Red candle or Doji
        lower_wick = close_p - low_p
        upper_wick = high_p - open_p
        body = open_p - close_p

    if body == 0: # Doji is not considered a pin bar for this rule
        return False

    return (lower_wick / total_range >= PIN_BAR_WICK_RATIO and
            body / total_range <= PIN_BAR_BODY_RATIO and
            upper_wick < body) # Upper wick must be smaller than body

def _is_bearish_pin_bar(open_p: float, high_p: float, low_p: float, close_p: float) -> bool:
    """Detects a bearish pin bar based on pseudocode ratios."""
    total_range = high_p - low_p
    if total_range == 0:
        return False

    if close_p > open_p: # Green candle
        lower_wick = open_p - low_p
        upper_wick = high_p - close_p
        body = close_p - open_p
    else: # Red candle or Doji
        lower_wick = close_p - low_p
        upper_wick = high_p - open_p
        body = open_p - close_p
    
    if body == 0: # Doji is not considered a pin bar for this rule
        return False

    return (upper_wick / total_range >= PIN_BAR_WICK_RATIO and
            body / total_range <= PIN_BAR_BODY_RATIO and
            lower_wick < body) # Lower wick must be smaller than body

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract data into numpy arrays for efficient calculations
        closes = np.array([c.close for c in warm_candles])
        highs = np.array([c.high for c in warm_candles])
        lows = np.array([c.low for c in warm_candles])
        opens = np.array([c.open_price for c in warm_candles])
        volumes = np.array([c.volume for c in warm_candles])

        # --- Calculate Bollinger Bands ---
        sma_closes_full = _calculate_sma(closes, BB_PERIOD)
        stddev_closes_full = _calculate_stddev(closes, BB_PERIOD)
        
        # Check if enough data for BBs
        if len(sma_closes_full) == 0 or len(stddev_closes_full) == 0:
            continue

        upper_bb_full = sma_closes_full + (stddev_closes_full * BB_DEV)
        lower_bb_full = sma_closes_full - (stddev_closes_full * BB_DEV)
        
        # BB_WIDTH = ((UPPER_BB - LOWER_BB) / SMA) * 100
        bb_width_full = np.zeros_like(sma_closes_full)
        non_zero_sma_idx = sma_closes_full != 0
        bb_width_full[non_zero_sma_idx] = ((upper_bb_full[non_zero_sma_idx] - lower_bb_full[non_zero_sma_idx]) / sma_closes_full[non_zero_sma_idx]) * 100

        # --- Calculate Money Flow Index (MFI) ---
        mfi_values_full = _calculate_mfi(highs, lows, closes, volumes, MFI_PERIOD)
        # Check if enough data for MFI and its reversal check (at least 3 values)
        if len(mfi_values_full) < 3:
            continue
        
        # --- Calculate Volume Moving Average ---
        avg_volume_full = _calculate_sma(volumes, VOLUME_MA_PERIOD)
        if len(avg_volume_full) == 0:
            continue

        # --- Calculate Bollinger Band Width Moving Average ---
        avg_bb_width_full = _calculate_sma(bb_width_full, BBW_MA_PERIOD)
        if len(avg_bb_width_full) == 0:
            continue

        # --- Extract latest values for the current candle (warm_candles[-1]) ---
        current_candle = warm_candles[-1]
        
        current_close = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        current_open = opens[-1]
        current_volume = volumes[-1]
        
        # The last element of each full indicator array corresponds to the latest candle
        current_upper_bb = upper_bb_full[-1]
        current_lower_bb = lower_bb_full[-1]
        current_bb_width = bb_width_full[-1]
        
        current_mfi = mfi_values_full[-1]
        prev_mfi = mfi_values_full[-2]
        prev2_mfi = mfi_values_full[-3] # MFI[2] in pseudocode

        current_avg_volume = avg_volume_full[-1]
        current_avg_bb_width = avg_bb_width_full[-1]

        # --- MFI Reversal Check ---
        mfi_turning_up = (current_mfi > prev_mfi) and (prev_mfi < prev2_mfi)
        mfi_turning_down = (current_mfi < prev_mfi) and (prev_mfi > prev2_mfi)

        # --- Candlestick Pattern Detection (Pin Bar) ---
        is_bullish_pin_bar = _is_bullish_pin_bar(current_open, current_high, current_low, current_close)
        is_bearish_pin_bar = _is_bearish_pin_bar(current_open, current_high, current_low, current_close)

        # --- Buy Signal ---
        if (is_bullish_pin_bar and
            current_low <= current_lower_bb and # Pin bar low touches or breaches lower BB
            current_close > current_lower_bb and # Pin bar closes back inside or above lower BB
            current_mfi < MFI_OVERSOLD and
            mfi_turning_up and
            current_volume > current_avg_volume and
            current_bb_width > current_avg_bb_width):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id="a2be20f2-122d-4065-8fa2-ca992ba2ae56",
                confidence=1.0 # Placeholder confidence
            ))

        # --- Sell Signal ---
        if (is_bearish_pin_bar and
            current_high >= current_upper_bb and # Pin bar high touches or breaches upper BB
            current_close < current_upper_bb and # Pin bar closes back inside or below upper BB
            current_mfi > MFI_OVERBOUGHT and
            mfi_turning_down and
            current_volume > current_avg_volume and
            current_bb_width > current_avg_bb_width):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id="a2be20f2-122d-4065-8fa2-ca992ba2ae56",
                confidence=1.0 # Placeholder confidence
            ))

    return signals