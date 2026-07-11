from __future__ import annotations
import numpy as np
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule Parameters ---
BB_PERIOD = 20
BB_STD_DEV = 2.0
MFI_PERIOD = 14
MFI_OVERSOLD = 20
MFI_OVERBOUGHT = 80
VOLUME_MA_PERIOD = 20
VOLUME_MULTIPLIER = 1.2
ATR_PERIOD = 14
ATR_THRESHOLD_PERCENT = 0.005  # ATR must be > 0.5% of the average price
MFI_DIV_LOOKBACK = 5  # Number of previous candles to check for MFI divergence

# Minimum candles required for all indicators + divergence + pattern
# This ensures enough data for the longest lookback period, and for MFI divergence.
# MFI divergence needs MFI_PERIOD candles to get the first MFI value, plus MFI_DIV_LOOKBACK
# additional candles to compare against.
MIN_CANDLES_REQUIRED = max(BB_PERIOD, MFI_PERIOD + MFI_DIV_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD)


# --- Helper Functions for Indicators ---

def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < period:
        return np.array([])
    return np.convolve(data, np.ones(period)/period, mode='valid')

def _calculate_std_dev(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Rolling Standard Deviation (population std dev)."""
    if len(data) < period:
        return np.array([])
    # Use ddof=0 for population standard deviation (common in TA)
    return np.array([np.std(data[i-period:i], ddof=0) for i in range(period, len(data) + 1)])

def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculates Average True Range (using SMA for smoothing)."""
    if len(close) < period:
        return np.array([])
    
    tr = np.zeros(len(close))
    for i in range(len(close)):
        h_l = high[i] - low[i]
        if i == 0:
            tr[i] = h_l
        else:
            h_pc = abs(high[i] - close[i-1])
            l_pc = abs(low[i] - close[i-1])
            tr[i] = max(h_l, h_pc, l_pc)
            
    # Use SMA for ATR as per other indicators in pseudocode
    atr_values = _calculate_sma(tr, period)
    return atr_values

def _calculate_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Calculates Money Flow Index."""
    if len(close) < period:
        return np.array([])
    
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_money_flow = np.zeros(len(close))
    negative_money_flow = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if typical_price[i] > typical_price[i-1]:
            positive_money_flow[i] = money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_money_flow[i] = money_flow[i]
            
    mfi_values = np.zeros(len(close) - period + 1)
    for i in range(len(mfi_values)):
        period_positive_mf = np.sum(positive_money_flow[i : i + period])
        period_negative_mf = np.sum(negative_money_flow[i : i + period])
        
        if period_negative_mf == 0:
            mfi_values[i] = 100.0 # Avoid division by zero, strong bullish
        else:
            money_ratio = period_positive_mf / period_negative_mf
            mfi_values[i] = 100 - (100 / (1 + money_ratio))
            
    return mfi_values

# --- Candlestick Pattern Detection ---
# Simplified general rejection pattern:
# A candle is a rejection if it touches/crosses the band,
# has a long wick in the direction of rejection,
# and closes away from the extreme.

def _is_bullish_rejection(current_candle: WarmCandle, lower_band: float) -> bool:
    """Checks for a bullish rejection pattern at or below the lower band."""
    if current_candle.low > lower_band: # Must touch or cross the lower band
        return False

    # A strong bullish close from the low, indicating rejection
    # Close should be above open (bullish candle)
    if current_candle.close > current_candle.open_price:
        # Check for a significant lower wick and close in upper part of range
        body_size = current_candle.close - current_candle.open_price
        total_range = current_candle.high - current_candle.low
        lower_wick = current_candle.open_price - current_candle.low
        
        # Condition 1: Lower wick is substantial (e.g., > 30% of total range)
        # Condition 2: Body is not excessively large (e.g., less than 50% of total range)
        # Condition 3: Close is in the upper half of the candle's range
        if total_range > 0 and lower_wick / total_range > 0.3 and body_size / total_range < 0.5:
            if current_candle.close >= current_candle.low + (total_range * 0.6): # Close in upper 40%
                return True
        
        # Alternative: Just a strong bullish candle that closed significantly above the lower band
        # after touching it, implying rejection.
        if current_candle.close > current_candle.open_price * 1.005 and current_candle.close > lower_band:
            return True
            
    return False

def _is_bearish_rejection(current_candle: WarmCandle, upper_band: float) -> bool:
    """Checks for a bearish rejection pattern at or above the upper band."""
    if current_candle.high < upper_band: # Must touch or cross the upper band
        return False

    # A strong bearish close from the high, indicating rejection
    # Close should be below open (bearish candle)
    if current_candle.close < current_candle.open_price:
        # Check for a significant upper wick and close in lower part of range
        body_size = current_candle.open_price - current_candle.close
        total_range = current_candle.high - current_candle.low
        upper_wick = current_candle.high - current_candle.open_price

        # Condition 1: Upper wick is substantial (e.g., > 30% of total range)
        # Condition 2: Body is not excessively large (e.g., less than 50% of total range)
        # Condition 3: Close is in the lower half of the candle's range
        if total_range > 0 and upper_wick / total_range > 0.3 and body_size / total_range < 0.5:
            if current_candle.close <= current_candle.high - (total_range * 0.6): # Close in lower 40%
                return True

        # Alternative: Just a strong bearish candle that closed significantly below the upper band
        # after touching it, implying rejection.
        if current_candle.close < current_candle.open_price * 0.995 and current_candle.close < upper_band:
            return True

    return False

# --- MFI Divergence Detection ---
def _has_bullish_mfi_divergence(
    mfi_values: np.ndarray, 
    low_prices: np.ndarray, 
    current_mfi: float, 
    current_low: float, 
    lookback_period: int, 
    oversold_threshold: float
) -> bool:
    """Detects bullish MFI divergence (price lower lows, MFI higher lows)."""
    if len(mfi_values) < lookback_period + 1 or len(low_prices) < lookback_period + 1:
        return False
    
    if current_mfi > oversold_threshold: # Current MFI must be oversold
        return False

    # Iterate backwards to find a previous low in price and MFI
    for i in range(1, lookback_period + 1):
        prev_mfi = mfi_values[-1 - i]
        prev_low = low_prices[-1 - i]
        
        # Check for MFI making higher lows while price makes lower lows
        if current_low < prev_low and current_mfi > prev_mfi:
            return True
            
    return False

def _has_bearish_mfi_divergence(
    mfi_values: np.ndarray, 
    high_prices: np.ndarray, 
    current_mfi: float, 
    current_high: float, 
    lookback_period: int, 
    overbought_threshold: float
) -> bool:
    """Detects bearish MFI divergence (price higher highs, MFI lower highs)."""
    if len(mfi_values) < lookback_period + 1 or len(high_prices) < lookback_period + 1:
        return False
    
    if current_mfi < overbought_threshold: # Current MFI must be overbought
        return False

    # Iterate backwards to find a previous high in price and MFI
    for i in range(1, lookback_period + 1):
        prev_mfi = mfi_values[-1 - i]
        prev_high = high_prices[-1 - i]
        
        # Check for MFI making lower highs while price makes higher highs
        if current_high > prev_high and current_mfi < prev_mfi:
            return True
            
    return False


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        
        # Ensure enough data for all calculations
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract data into numpy arrays for efficient calculation
        closes = np.array([c.close for c in warm_candles])
        highs = np.array([c.high for c in warm_candles])
        lows = np.array([c.low for c in warm_candles])
        opens = np.array([c.open_price for c in warm_candles])
        volumes = np.array([c.volume for c in warm_candles])
        
        current_candle = warm_candles[-1] # The latest complete candle

        # --- Calculate Indicators ---
        
        # Bollinger Bands
        bb_sma = _calculate_sma(closes, BB_PERIOD)
        bb_std_dev = _calculate_std_dev(closes, BB_PERIOD)

        if len(bb_sma) < 1 or len(bb_std_dev) < 1: # Should not happen if MIN_CANDLES_REQUIRED is met
            continue
            
        upper_band = bb_sma[-1] + (bb_std_dev[-1] * BB_STD_DEV)
        lower_band = bb_sma[-1] - (bb_std_dev[-1] * BB_STD_DEV)

        # MFI
        mfi_values = _calculate_mfi(highs, lows, closes, volumes, MFI_PERIOD)
        if len(mfi_values) < 1: # Should not happen if MIN_CANDLES_REQUIRED is met
            continue
        current_mfi = mfi_values[-1]

        # Average Volume
        avg_volume_values = _calculate_sma(volumes, VOLUME_MA_PERIOD)
        if len(avg_volume_values) < 1: # Should not happen if MIN_CANDLES_REQUIRED is met
            continue
        current_avg_volume = avg_volume_values[-1]
        
        # ATR
        atr_values = _calculate_atr(highs, lows, closes, ATR_PERIOD)
        if len(atr_values) < 1: # Should not happen if MIN_CANDLES_REQUIRED is met
            continue
        current_atr = atr_values[-1]

        # --- Check Conditions ---
        
        # Candlestick Rejection
        is_bullish_rejection = _is_bullish_rejection(current_candle, lower_band)
        is_bearish_rejection = _is_bearish_rejection(current_candle, upper_band)

        # MFI Divergence
        # The MFI values are calculated from MFI_PERIOD candles.
        # The `mfi_values` array's index `k` corresponds to the candle at `warm_candles[k + MFI_PERIOD - 1]`.
        # So, to align prices with MFI values for divergence check, we slice the original price arrays.
        mfi_aligned_lows = lows[MFI_PERIOD - 1:]
        mfi_aligned_highs = highs[MFI_PERIOD - 1:]

        has_bullish_mfi_divergence = _has_bullish_mfi_divergence(
            mfi_values=mfi_values,
            low_prices=mfi_aligned_lows,
            current_mfi=current_mfi,
            current_low=current_candle.low,
            lookback_period=MFI_DIV_LOOKBACK,
            oversold_threshold=MFI_OVERSOLD
        )
        has_bearish_mfi_divergence = _has_bearish_mfi_divergence(
            mfi_values=mfi_values,
            high_prices=mfi_aligned_highs,
            current_mfi=current_mfi,
            current_high=current_candle.high,
            lookback_period=MFI_DIV_LOOKBACK,
            overbought_threshold=MFI_OVERBOUGHT
        )

        # Volume Confirmation
        current_volume = current_candle.volume
        is_volume_confirmed = current_volume > (current_avg_volume * VOLUME_MULTIPLIER)

        # Volatility Filter
        # ATR threshold is a percentage of the average close price over the ATR period.
        # Ensure enough data for the average_price for ATR threshold.
        if len(closes) < ATR_PERIOD: # Should already be covered by MIN_CANDLES_REQUIRED
            continue
        average_price_for_atr_threshold = np.mean(closes[-ATR_PERIOD:])
        atr_threshold_value = average_price_for_atr_threshold * ATR_THRESHOLD_PERCENT
        is_volatility_confirmed = current_atr > atr_threshold_value

        # --- Generate Signals ---
        
        # Buy Signal Conditions
        if (is_bullish_rejection and 
            current_mfi < MFI_OVERSOLD and 
            has_bullish_mfi_divergence and 
            is_volume_confirmed and 
            is_volatility_confirmed):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id="7f887817-acfa-468c-8726-7d67494fcb86"
            ))

        # Sell Signal Conditions
        if (is_bearish_rejection and 
            current_mfi > MFI_OVERBOUGHT and 
            has_bearish_mfi_divergence and 
            is_volume_confirmed and 
            is_volatility_confirmed):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id="7f887817-acfa-468c-8726-7d67494fcb86"
            ))

    return signals