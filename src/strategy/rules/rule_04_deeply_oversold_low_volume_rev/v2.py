from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, WarmCandle, SellSignal

# --- Helper Functions for Indicator Calculations ---

def _calculate_sma(candles: list[WarmCandle], period: int, price_type: str) -> float:
    """Calculates the Simple Moving Average (SMA) for a given price type over a specified period."""
    if len(candles) < period:
        raise ValueError(f"Insufficient candles ({len(candles)}) for SMA period {period}.")
    
    # Extract the relevant prices/volumes from the last 'period' candles
    if price_type == 'close':
        values = [c.close for c in candles[-period:]]
    elif price_type == 'volume':
        values = [c.volume for c in candles[-period:]]
    else:
        raise ValueError(f"Unsupported price_type for SMA: {price_type}")
        
    return statistics.mean(values)

def _calculate_ema(candles: list[WarmCandle], period: int) -> float:
    """Calculates the Exponential Moving Average (EMA) for close prices."""
    if len(candles) < period:
        raise ValueError(f"Insufficient candles ({len(candles)}) for EMA period {period}.")

    alpha = 2 / (period + 1)
    
    # Calculate initial SMA for the first EMA value
    # Use the first 'period' candles for the initial EMA value
    ema = sum(c.close for c in candles[:period]) / period

    # Apply EMA formula for the rest of the candles, including the last one
    for i in range(period, len(candles)):
        ema = (candles[i].close * alpha) + (ema * (1 - alpha))
        
    return ema

def _calculate_atr(candles: list[WarmCandle], period: int) -> float:
    """Calculates the Average True Range (ATR) using Wilder's smoothing method."""
    # Need period + 1 candles for the first ATR value:
    #   1 candle for the first True Range (requires prev_close)
    #   'period' candles for the initial SMA of True Ranges
    if len(candles) < period + 1:
        raise ValueError(f"Insufficient candles ({len(candles)}) for ATR period {period}. Need at least {period + 1}.")

    true_ranges = []
    # Calculate True Ranges for all available candles (from the second candle onwards)
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i-1].close
        
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr) # true_ranges will have len(candles) - 1 elements

    # Ensure we have enough true_ranges to calculate the initial average
    if len(true_ranges) < period:
        # This condition should ideally be caught by the initial len(candles) check
        raise ValueError(f"Internal error: Not enough true ranges ({len(true_ranges)}) for ATR period {period} after calculation.")

    # Calculate the initial ATR as a simple average of the first 'period' True Ranges
    current_atr = sum(true_ranges[:period]) / period

    # Apply Wilder's smoothing for subsequent ATR values
    for i in range(period, len(true_ranges)):
        current_atr = ((current_atr * (period - 1)) + true_ranges[i]) / period
        
    return current_atr

def _calculate_rsi(candles: list[WarmCandle], period: int) -> float:
    """Calculates the Relative Strength Index (RSI) using Wilder's smoothing method."""
    # Need period + 1 candles for the first RSI value:
    #   1 candle for the first price change calculation
    #   'period' candles for the initial average gain/loss
    if len(candles) < period + 1:
        raise ValueError(f"Insufficient candles ({len(candles)}) for RSI period {period}. Need at least {period + 1}.")

    gains = []
    losses = []
    
    # Calculate price changes, separating into gains and losses
    for i in range(1, len(candles)):
        change = candles[i].close - candles[i-1].close
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))

    # Ensure we have enough changes to calculate the initial average gain and loss
    if len(gains) < period:
        # This condition should ideally be caught by the initial len(candles) check
        raise ValueError(f"Internal error: Not enough changes ({len(gains)}) for RSI period {period} after calculation.")

    # Calculate initial average gain and loss over the first 'period' changes
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Apply Wilder's smoothing for subsequent average gains and losses
    for i in range(period, len(gains)): # Iterate through the remaining changes
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    # Handle division by zero for RS calculation
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0 # If no losses, RSI is 100. If no changes, RSI is 50.
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Rule Configuration ---

RSI_PERIOD = 14
KELTNER_EMA_PERIOD = 20 # Keltner Middle Band uses EMA as per pseudocode
KELTNER_ATR_PERIOD = 20 # Keltner ATR period as per pseudocode
KELTNER_MULTIPLIER = 2.0 
VOLUME_ANOMALY_PERIOD = 20 # Volume SMA period for anomaly as per pseudocode

# Minimum number of candles required for all indicators:
# RSI(14) needs 14+1 = 15 candles.
# EMA(20) needs 20 candles.
# ATR(20) needs 20+1 = 21 candles.
# SMA(20) for volume needs 20 candles.
MIN_CANDLES_REQUIRED = max(RSI_PERIOD + 1, KELTNER_EMA_PERIOD, KELTNER_ATR_PERIOD + 1, VOLUME_ANOMALY_PERIOD) # = 21

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Extreme Oversold Mean Reversion with Constrained Low Volume' strategy.
    Identifies potential mean-reversion buy opportunities when an asset is deeply oversold
    (low RSI, significantly below Keltner Channel) combined with a specific range of low trading volume.
    """
    signals: list[BuySignal | SellSignal] = []
    
    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # 0. Check for sufficient data
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue # Skip this pair due to insufficient data

        latest_candle = warm_candles[-1]
        latest_close = latest_candle.close
        latest_volume = latest_candle.volume
        latest_timestamp = latest_candle.hour
        
        try:
            # 1. Calculate Hourly RSI(14)
            latest_rsi = _calculate_rsi(warm_candles, period=RSI_PERIOD)

            # 2. Calculate Hourly Keltner Channel Position
            middle_band = _calculate_ema(warm_candles, period=KELTNER_EMA_PERIOD) # Using EMA for middle band
            atr_value = _calculate_atr(warm_candles, period=KELTNER_ATR_PERIOD) # Using ATR period 20
            
            # Keltner Channel Position = (latest_close - middle_band) / (Multiplier * ATR)
            # Handle division by zero if ATR is 0
            if atr_value == 0:
                keltner_channel_position = float('-inf') if latest_close < middle_band else \
                                           float('inf') if latest_close > middle_band else \
                                           0.0
            else:
                keltner_channel_position = (latest_close - middle_band) / (KELTNER_MULTIPLIER * atr_value)

            # 3. Calculate Hourly Volume Anomaly
            avg_volume_period = _calculate_sma(warm_candles, period=VOLUME_ANOMALY_PERIOD, price_type='volume')
            
            volume_anomaly = 0.0 # Default if average volume is zero
            if avg_volume_period > 0:
                volume_anomaly = latest_volume / avg_volume_period
            # If avg_volume_period is 0, volume_anomaly remains 0.0.
            # If latest_volume is also 0, this fits the 0.2 <= anomaly <= 0.7 range.
            # If avg_volume_period is 0 but latest_volume > 0, anomaly would be inf, which would not satisfy the condition.

            # BUY Signal Conditions:
            # RSI below 30
            # Keltner Channel Position below -1.0
            # Volume Anomaly strictly between 0.2 and 0.7
            if (latest_rsi < 30 and
                keltner_channel_position < -1.0 and
                0.2 <= volume_anomaly <= 0.7):
                
                signals.append(BuySignal(
                    pair=pair,
                    timestamp=latest_timestamp,
                    price=latest_close,
                    rule_id="extreme_oversold_mr_v2", # Matches idea_id
                    confidence=1.0 
                ))

        except ValueError:
            # Catch errors from indicator calculations due to specific data issues (e.g., all zeros, insufficient data for a specific period within the general check)
            # and skip this pair. Insufficient data for length is already handled.
            continue 

    return signals