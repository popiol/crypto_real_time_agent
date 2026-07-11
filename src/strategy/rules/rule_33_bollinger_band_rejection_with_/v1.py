from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule ID ---
RULE_ID = "rule_362bfd3b-881b-45db-9823-26a49fc0f757"

# --- Parameters ---
BB_PERIOD = 20
BB_STDDEV = 2
MFI_PERIOD = 14
MFI_OVERSOLD = 20
MFI_OVERBOUGHT = 80
VOLUME_SMA_PERIOD = 20
VOLUME_MULTIPLIER = 1.2

# Minimum number of candles required for all calculations
# MFI needs period + 1 for typical price comparison.
# MFI turn needs at least two MFI values.
# Candlestick patterns need current and previous candle.
MIN_CANDLES_REQUIRED = max(BB_PERIOD, MFI_PERIOD + 1, VOLUME_SMA_PERIOD) + 1


# --- Helper Functions for Indicators ---

def calculate_bollinger_bands(closes: np.ndarray, period: int, stddev: float) -> tuple[float, float, float]:
    """Calculates Bollinger Bands (Middle, Upper, Lower)."""
    if len(closes) < period:
        return np.nan, np.nan, np.nan
    
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    
    upper = sma + std * stddev
    lower = sma - std * stddev
    
    return sma, upper, lower

def calculate_mfi(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int
) -> float:
    """Calculates Money Flow Index (MFI)."""
    # Requires at least 'period' data points for the sum, plus one for the initial typical price comparison.
    # So, need 'period + 1' candles to calculate the MFI for the last candle.
    if len(highs) < period + 1:
        return np.nan

    typical_prices = (highs + lows + closes) / 3
    
    raw_money_flow = typical_prices * volumes

    positive_money_flow = np.zeros_like(raw_money_flow)
    negative_money_flow = np.zeros_like(raw_money_flow)

    # Calculate positive and negative money flow for each period
    for i in range(1, len(typical_prices)):
        if typical_prices[i] > typical_prices[i-1]:
            positive_money_flow[i] = raw_money_flow[i]
        elif typical_prices[i] < typical_prices[i-1]:
            negative_money_flow[i] = raw_money_flow[i]

    # Sum over the last 'period' candles for positive and negative money flow
    period_positive_money_flow = np.sum(positive_money_flow[-period:])
    period_negative_money_flow = np.sum(negative_money_flow[-period:])

    if period_negative_money_flow == 0:
        if period_positive_money_flow == 0:
            return 50.0 # Neutral if no money flow in either direction
        return 100.0 # Strong upward pressure, MFI is 100
    
    money_ratio = period_positive_money_flow / period_negative_money_flow
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi

def calculate_volume_sma(volumes: np.ndarray, period: int) -> float:
    """Calculates Simple Moving Average of Volume."""
    if len(volumes) < period:
        return np.nan
    return np.mean(volumes[-period:])


# --- Candlestick Pattern Detection ---

def is_bullish_rejection(current_candle: WarmCandle, previous_candle: WarmCandle) -> bool:
    """
    Detects bullish reversal candlestick patterns (Hammer or Bullish Engulfing).
    Based on pseudocode for Hammer and common definition for Engulfing.
    """
    cc = current_candle # shorthand
    pc = previous_candle # shorthand

    # Avoid division by zero for range, use a small fraction of close price for robustness
    min_divisor = 0.001 * cc.close 
    
    # Hammer pattern conditions (from pseudocode)
    # (current_candle.close > current_candle.open AND (current_candle.high - current_candle.low) > 3 * (current_candle.open - current_candle.close) AND (current_candle.close - current_candle.low) / (0.001 + current_candle.high - current_candle.low) > 0.6)
    is_hammer = False
    if cc.close > cc.open_price: # Bullish candle
        body_size = cc.close - cc.open_price
        total_range = cc.high - cc.low
        lower_shadow_length = cc.open_price - cc.low
        
        if total_range > min_divisor: # Ensure total_range is not zero
            is_hammer = (
                total_range > 3 * body_size and 
                (lower_shadow_length / total_range) > 0.6
            )
    
    if is_hammer:
        return True

    # Bullish Engulfing pattern conditions
    is_bullish_engulfing = (
        pc.close < pc.open_price and # Previous candle is bearish
        cc.close > cc.open_price and # Current candle is bullish
        cc.open_price < pc.close and # Current candle opens below previous close
        cc.close > pc.open_price # Current candle closes above previous open
    )
    if is_bullish_engulfing:
        return True

    return False

def is_bearish_rejection(current_candle: WarmCandle, previous_candle: WarmCandle) -> bool:
    """
    Detects bearish reversal candlestick patterns (Shooting Star or Bearish Engulfing).
    Based on pseudocode for Shooting Star and common definition for Engulfing.
    """
    cc = current_candle # shorthand
    pc = previous_candle # shorthand

    min_divisor = 0.001 * cc.close # Avoid division by zero
    
    # Shooting Star pattern conditions (from pseudocode)
    # (current_candle.open > current_candle.close AND (current_candle.high - current_candle.low) > 3 * (current_candle.high - current_candle.open) AND (current_candle.high - current_candle.open) / (0.001 + current_candle.high - current_candle.low) > 0.6)
    is_shooting_star = False
    if cc.open_price > cc.close: # Bearish candle
        body_size = cc.open_price - cc.close
        total_range = cc.high - cc.low
        upper_shadow_length = cc.high - cc.open_price
        
        if total_range > min_divisor: # Ensure total_range is not zero
            is_shooting_star = (
                total_range > 3 * body_size and 
                (upper_shadow_length / total_range) > 0.6
            )
    
    if is_shooting_star:
        return True
    
    # Bearish Engulfing pattern conditions
    is_bearish_engulfing = (
        pc.close > pc.open_price and # Previous candle is bullish
        cc.close < cc.open_price and # Current candle is bearish
        cc.open_price > pc.close and # Current candle opens above previous close
        cc.close < pc.open_price # Current candle closes below previous open
    )
    if is_bearish_engulfing:
        return True

    return False


# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract necessary data as numpy arrays for efficient calculation
        closes = np.array([c.close for c in warm_candles])
        highs = np.array([c.high for c in warm_candles])
        lows = np.array([c.low for c in warm_candles])
        opens = np.array([c.open_price for c in warm_candles])
        volumes = np.array([c.volume for c in warm_candles])

        current_candle = warm_candles[-1]
        previous_candle = warm_candles[-2] # Needed for MFI turn and candlestick patterns

        # 1. Calculate Bollinger Bands
        _, bb_upper, bb_lower = calculate_bollinger_bands(closes, BB_PERIOD, BB_STDDEV)
        if np.isnan(bb_upper) or np.isnan(bb_lower):
            continue

        # 2. Calculate MFI
        # To check for MFI turn, we need MFI for the current and previous period.
        # We calculate MFI values for a window that includes the previous period's MFI calculation.
        
        # Slice for current MFI calculation (last MIN_CANDLES_REQUIRED data points)
        mfi_highs_slice = highs[-MFI_PERIOD - 1:]
        mfi_lows_slice = lows[-MFI_PERIOD - 1:]
        mfi_closes_slice = closes[-MFI_PERIOD - 1:]
        mfi_volumes_slice = volumes[-MFI_PERIOD - 1:]

        current_mfi = calculate_mfi(
            mfi_highs_slice,
            mfi_lows_slice,
            mfi_closes_slice,
            mfi_volumes_slice,
            MFI_PERIOD
        )
        
        # Slice for previous MFI calculation (data points ending one candle prior)
        prev_mfi_highs_slice = highs[-MFI_PERIOD - 2:-1]
        prev_mfi_lows_slice = lows[-MFI_PERIOD - 2:-1]
        prev_mfi_closes_slice = closes[-MFI_PERIOD - 2:-1]
        prev_mfi_volumes_slice = volumes[-MFI_PERIOD - 2:-1]

        previous_mfi = calculate_mfi(
            prev_mfi_highs_slice,
            prev_mfi_lows_slice,
            prev_mfi_closes_slice,
            prev_mfi_volumes_slice,
            MFI_PERIOD
        )
        
        if np.isnan(current_mfi) or np.isnan(previous_mfi):
            continue

        # 3. Calculate Volume SMA
        volume_sma = calculate_volume_sma(volumes, VOLUME_SMA_PERIOD)
        if np.isnan(volume_sma):
            continue

        # --- Buy Signal Conditions ---
        # IF (CURRENT_CANDLE.close < BB_LOWER AND
        #     MFI < MFI_OVERSOLD AND
        #     MFI > PREVIOUS_MFI AND // MFI turning upward
        #     CURRENT_VOLUME > (VOLUME_SMA * VOLUME_MULTIPLIER) AND
        #     IsBullishRejection(CURRENT_CANDLE, PREVIOUS_CANDLE)) THEN
        
        if (current_candle.close < bb_lower and
            current_mfi < MFI_OVERSOLD and
            current_mfi > previous_mfi and # MFI turning upward
            current_candle.volume > (volume_sma * VOLUME_MULTIPLIER) and
            is_bullish_rejection(current_candle, previous_candle)):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id=RULE_ID
            ))

        # --- Sell Signal Conditions ---
        # ELSE IF (CURRENT_CANDLE.close > BB_UPPER AND
        #          MFI > MFI_OVERBOUGHT AND
        #          MFI < PREVIOUS_MFI AND // MFI turning downward
        #          CURRENT_VOLUME > (VOLUME_SMA * VOLUME_MULTIPLIER) AND
        #          IsBearishRejection(CURRENT_CANDLE, PREVIOUS_CANDLE)) THEN

        elif (current_candle.close > bb_upper and
              current_mfi > MFI_OVERBOUGHT and
              current_mfi < previous_mfi and # MFI turning downward
              current_candle.volume > (volume_sma * VOLUME_MULTIPLIER) and
              is_bearish_rejection(current_candle, previous_candle)):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id=RULE_ID
            ))

    return signals