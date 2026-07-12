from __future__ import annotations
import statistics
import numpy as np
from datetime import datetime
from typing import List, Union

from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Constants ---
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2
MFI_PERIOD = 14
MFI_OVERSOLD = 20
MFI_OVERBOUGHT = 80
VOLUME_SMA_PERIOD = 20
BBW_SMA_PERIOD = 20

# Minimum number of candles required to calculate all indicators
# 1. BB and BBW for current & previous: BB_PERIOD + 1
# 2. MFI for current & previous: MFI_PERIOD + 1
# 3. Volume SMA for current: VOLUME_SMA_PERIOD
# 4. SMA of BBW for current: BB_PERIOD (for individual BBW values) + BBW_SMA_PERIOD - 1 (for SMA of those BBW values)
MIN_CANDLES_REQUIRED = max(
    BB_PERIOD + 1,
    MFI_PERIOD + 1,
    VOLUME_SMA_PERIOD,
    BB_PERIOD + BBW_SMA_PERIOD - 1
)


# --- Helper Functions for Indicators ---

def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < period:
        return np.array([])
    return np.convolve(data, np.ones(period) / period, mode='valid')

def calculate_bollinger_bands(closes: np.ndarray, period: int, std_dev_multiplier: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates Bollinger Bands (Middle, Upper, Lower) and Bollinger Band Width."""
    if len(closes) < period:
        return np.array([]), np.array([]), np.array([]), np.array([])

    sma = calculate_sma(closes, period)
    
    # Calculate rolling standard deviation
    std_devs = np.array([np.std(closes[i : i + period]) for i in range(len(closes) - period + 1)])

    upper_band = sma + (std_devs * std_dev_multiplier)
    lower_band = sma - (std_devs * std_dev_multiplier)
    
    # Bollinger Band Width: (Upper Band - Lower Band) / Middle Band
    bb_width = (upper_band - lower_band) / sma if len(sma) > 0 else np.array([])

    return sma, upper_band, lower_band, bb_width

def calculate_mfi(candles: List[WarmCandle], period: int) -> np.ndarray:
    """Calculates Money Flow Index."""
    if len(candles) < period:
        return np.array([])

    tps = np.array([(c.high + c.low + c.close) / 3 for c in candles])
    volumes = np.array([c.volume for c in candles])
    
    mfs = tps * volumes

    pmfs = np.zeros(len(candles))
    nmfs = np.zeros(len(candles))

    for i in range(1, len(candles)):
        if tps[i] > tps[i-1]:
            pmfs[i] = mfs[i]
        elif tps[i] < tps[i-1]:
            nmfs[i] = mfs[i]
    
    mfi_values = np.full(len(candles) - period + 1, np.nan)

    for i in range(len(mfi_values)):
        current_pmf_sum = np.sum(pmfs[i : i + period])
        current_nmf_sum = np.sum(nmfs[i : i + period])
        
        if current_nmf_sum == 0:
            mfi_values[i] = 100 if current_pmf_sum > 0 else 50
        else:
            money_ratio = current_pmf_sum / current_nmf_sum
            mfi_values[i] = 100 - (100 / (1 + money_ratio))
            
    return mfi_values


# --- Candlestick Pattern Detection ---

def is_bullish_engulfing(prev_c: WarmCandle, curr_c: WarmCandle) -> bool:
    """Checks for a Bullish Engulfing pattern."""
    # Previous candle must be bearish
    if prev_c.close >= prev_c.open:
        return False
    # Current candle must be bullish
    if curr_c.close <= curr_c.open:
        return False
    # Current candle's body must engulf previous candle's body
    return curr_c.open < prev_c.close and curr_c.close > prev_c.open

def is_bearish_engulfing(prev_c: WarmCandle, curr_c: WarmCandle) -> bool:
    """Checks for a Bearish Engulfing pattern."""
    # Previous candle must be bullish
    if prev_c.close <= prev_c.open:
        return False
    # Current candle must be bearish
    if curr_c.close >= curr_c.open:
        return False
    # Current candle's body must engulf previous candle's body
    return curr_c.open > prev_c.close and curr_c.close < prev_c.open


# --- Main Signal Function ---

def signal(data: MarketData) -> List[Union[BuySignal, SellSignal]]:
    signals: List[Union[BuySignal, SellSignal]] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm
        
        if len(candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract necessary data for numpy calculations
        closes = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])

        # --- Calculate Indicators ---
        
        bb_middle_bands, bb_upper_bands, bb_lower_bands, bb_widths = calculate_bollinger_bands(
            closes, BB_PERIOD, BB_STD_DEV_MULTIPLIER
        )
        
        # MFI
        mfi_values = calculate_mfi(candles, MFI_PERIOD)

        # SMA of Volume
        volume_smas = calculate_sma(volumes, VOLUME_SMA_PERIOD)
        
        # SMA of BBW
        sma_bb_widths = calculate_sma(bb_widths, BBW_SMA_PERIOD)

        # Ensure all calculated indicator arrays have enough data points for the latest candles
        # These checks are redundant if MIN_CANDLES_REQUIRED is correctly calculated and applied
        # and the helper functions return empty arrays for insufficient data.
        # However, they add robustness against potential edge cases or future changes in helper logic.
        if len(bb_upper_bands) < 2 or len(mfi_values) < 2 or len(volume_smas) < 1 or len(sma_bb_widths) < 1:
            continue

        # Get current and previous candles
        current_candle = candles[-1]
        previous_candle = candles[-2]

        # Indicator values for current_candle (index -1 in indicator arrays)
        current_bb_upper = bb_upper_bands[-1]
        current_bb_lower = bb_lower_bands[-1]
        current_bb_width = bb_widths[-1]
        current_mfi = mfi_values[-1]
        current_volume_sma = volume_smas[-1]
        current_sma_bb_width = sma_bb_widths[-1]

        # Indicator values for previous_candle (index -2 in indicator arrays)
        previous_bb_upper = bb_upper_bands[-2]
        previous_bb_lower = bb_lower_bands[-2]
        previous_mfi = mfi_values[-2]

        # --- Buy Signal Conditions ---
        # 1. previous_candle.close < BB_Lower_Band (previous candle breached the lower band).
        # 2. current_candle forms a Bullish Engulfing pattern:
        #    - current_candle.open < previous_candle.close
        #    - current_candle.close > previous_candle.open
        #    - current_candle.close > BB_Lower_Band (engulfing candle closes back inside the band).
        # 3. MFI on current_candle is oversold (e.g., < 20) AND MFI(current_candle) > MFI(previous_candle) (MFI turning upward).
        # 4. current_candle.volume > SMA_Volume(current_candle).
        # 5. BBW(current_candle) > SMA_BBW(current_candle) (higher volatility).

        if (previous_candle.close < previous_bb_lower and
            is_bullish_engulfing(previous_candle, current_candle) and
            current_candle.close > current_bb_lower and 
            current_mfi < MFI_OVERSOLD and
            current_mfi > previous_mfi and
            current_candle.volume > current_volume_sma and
            current_bb_width > current_sma_bb_width):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id="389ebce7-c34c-4b26-aa19-e2614ceb5368",
                confidence=1.0 
            ))

        # --- Sell Signal Conditions ---
        # 1. previous_candle.close > BB_Upper_Band (previous candle breached the upper band).
        # 2. current_candle forms a Bearish Engulfing pattern:
        #    - current_candle.open > previous_candle.close
        #    - current_candle.close < previous_candle.open
        #    - current_candle.close < BB_Upper_Band (engulfing candle closes back inside the band).
        # 3. MFI on current_candle is overbought (e.g., > 80) AND MFI(current_candle) < MFI(previous_candle) (MFI turning downward).
        # 4. current_candle.volume > SMA_Volume(current_candle).
        # 5. BBW(current_candle) > SMA_BBW(current_candle) (higher volatility).
        
        elif (previous_candle.close > previous_bb_upper and
              is_bearish_engulfing(previous_candle, current_candle) and
              current_candle.close < current_bb_upper and 
              current_mfi > MFI_OVERBOUGHT and
              current_mfi < previous_mfi and
              current_candle.volume > current_volume_sma and
              current_bb_width > current_sma_bb_width):

            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id="389ebce7-c34c-4b26-aa19-e2614ceb5368",
                confidence=1.0
            ))

    return signals