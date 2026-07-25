from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, WarmCandle

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
SMA_MIDDLE_BAND_PERIOD = 20
ATR_PERIOD = 10
VOLUME_SMA_PERIOD = 10
KELTNER_MULTIPLIER = 2.0 # Keltner Channel is typically defined with 2 * ATR for bands.

# Minimum number of candles required for all indicators:
# RSI(14) needs 14+1 = 15 candles.
# SMA(20) needs 20 candles.
# ATR(10) needs 10+1 = 11 candles.
# SMA(10) for volume needs 10 candles.
# The maximum of these is 20, which aligns with the pseudocode's `len(data.warm) < 20`.
MIN_CANDLES_REQUIRED = max(RSI_PERIOD + 1, SMA_MIDDLE_BAND_PERIOD, ATR_PERIOD + 1, VOLUME_SMA_PERIOD)

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal]:
    """
    Implements the 'Deeply Oversold Low Volume Reversal Buy' strategy.
    Identifies potential mean-reversion buy opportunities when an asset is deeply oversold
    (low RSI, significantly below Keltner Channel) combined with unusually low trading volume.
    """
    signals: list[BuySignal] = []
    
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
            middle_band = _calculate_sma(warm_candles, period=SMA_MIDDLE_BAND_PERIOD, price_type='close')
            atr_value = _calculate_atr(warm_candles, period=ATR_PERIOD)
            
            # Keltner Channel Position = (latest_close - middle_band) / (Multiplier * ATR)
            if atr_value == 0:
                # If ATR is zero, the channel width is zero.
                # If current price is below the middle band, it's infinitely far below.
                keltner_channel_position = float('-inf') if latest_close < middle_band else \
                                           float('inf') if latest_close > middle_band else \
                                           0.0 # Exactly on the band
            else:
                keltner_channel_position = (latest_close - middle_band) / (KELTNER_MULTIPLIER * atr_value)

            # 3. Calculate Hourly Volume Anomaly
            avg_volume_10_period = _calculate_sma(warm_candles, period=VOLUME_SMA_PERIOD, price_type='volume')
            
            volume_anomaly = 0.0 # Default if average volume is zero
            if avg_volume_10_period > 0:
                volume_anomaly = latest_volume / avg_volume_10_period
            else:
                # If average volume is zero, and current volume is also zero, anomaly is 0.
                # If current volume is > 0 and average is 0, it's an infinite anomaly,
                # but for this rule, we are looking for *low* volume anomaly.
                # So, if avg_volume_10_period is 0, we treat it as extremely low activity,
                # satisfying the low volume condition (0.0 < 0.7).
                pass # volume_anomaly remains 0.0

            # BUY Signal Conditions:
            if (latest_rsi < 30 and
                keltner_channel_position < -0.8 and
                volume_anomaly < 0.7):
                
                signals.append(BuySignal(
                    pair=pair,
                    timestamp=latest_timestamp,
                    price=latest_close,
                    rule_id="deeply_oversold_low_vol_rebound", # Matches idea_id from prompt
                    confidence=1.0 # Placeholder confidence
                ))

        except ValueError:
            # Catch errors from indicator calculations due to specific data issues (e.g., all zeros)
            # and skip this pair. Insufficient data for length is already handled.
            continue 

    return signals