from __future__ import annotations
import numpy as np
import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Constants ---
PERIOD_BB = 20
STD_DEV_BB = 2
PERIOD_MFI = 14
PERIOD_VOLUME_SMA = 20
MFI_OVERSOLD_THRESHOLD = 20
MFI_OVERBOUGHT_THRESHOLD = 80
VOLUME_THRESHOLD_MULTIPLIER = 1.5

# Minimum number of candles required for calculations:
# - Bollinger Bands: PERIOD_BB
# - Money Flow Index: PERIOD_MFI + 1 for current MFI, and PERIOD_MFI + 2 for current and previous MFI
# - Volume SMA: PERIOD_VOLUME_SMA
MIN_CANDLES = max(PERIOD_BB, PERIOD_MFI + 2, PERIOD_VOLUME_SMA)

def _calculate_bollinger_bands(close_prices: np.ndarray) -> tuple[float, float, float]:
    """
    Calculates Bollinger Bands (SMA, Upper, Lower) for the last data point
    based on the last PERIOD_BB close prices.
    """
    if len(close_prices) < PERIOD_BB:
        return np.nan, np.nan, np.nan

    relevant_prices = close_prices[-PERIOD_BB:]
    
    sma = np.mean(relevant_prices)
    std_dev = np.std(relevant_prices)
    
    upper_bb = sma + (STD_DEV_BB * std_dev)
    lower_bb = sma - (STD_DEV_BB * std_dev)
    
    return sma, upper_bb, lower_bb

def _mfi_single_calc(tp_arr: np.ndarray, vol_arr: np.ndarray) -> float:
    """
    Calculates the Money Flow Index (MFI) for a given window of typical prices and volumes.
    Requires at least 2 data points for calculation.
    """
    positive_money_flow = 0.0
    negative_money_flow = 0.0

    if len(tp_arr) < 2:
        return np.nan

    for i in range(1, len(tp_arr)):
        current_tp = tp_arr[i]
        prev_tp = tp_arr[i-1]
        
        # Money Flow for the current candle (Typical Price * Volume)
        money_flow = current_tp * vol_arr[i] 

        if current_tp > prev_tp:
            positive_money_flow += money_flow
        elif current_tp < prev_tp:
            negative_money_flow += money_flow
        # If current_tp == prev_tp, money flow is not added to positive or negative.

    if negative_money_flow == 0:
        if positive_money_flow == 0:
            return 50.0 # No price change over the period, MFI is typically 50
        return 100.0 # Avoid division by zero, indicates strong positive money flow
    
    money_ratio = positive_money_flow / negative_money_flow
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

def _calculate_mfi(candles: list[WarmCandle]) -> tuple[float, float]:
    """
    Calculates the current and previous MFI values.
    Requires at least PERIOD_MFI + 2 candles for both current and previous MFI.
    """
    if len(candles) < PERIOD_MFI + 2:
        return np.nan, np.nan

    typical_prices = np.array([(c.high + c.low + c.close) / 3 for c in candles])
    volumes = np.array([c.volume for c in candles])

    # Calculate MFI for the current period
    # This requires PERIOD_MFI + 1 candles to get PERIOD_MFI price changes
    current_mfi_tp = typical_prices[-(PERIOD_MFI + 1):]
    current_mfi_vol = volumes[-(PERIOD_MFI + 1):]
    current_mfi = _mfi_single_calc(current_mfi_tp, current_mfi_vol)

    # Calculate MFI for the previous period (window shifted back by one candle)
    # This requires candles from `-(PERIOD_MFI + 2)` up to `-1`
    previous_mfi_tp = typical_prices[-(PERIOD_MFI + 2):-1]
    previous_mfi_vol = volumes[-(PERIOD_MFI + 2):-1]
    previous_mfi = _mfi_single_calc(previous_mfi_tp, previous_mfi_vol)

    return current_mfi, previous_mfi

def _calculate_volume_sma(volumes: np.ndarray) -> float:
    """
    Calculates the Simple Moving Average of volume for the last PERIOD_VOLUME_SMA periods.
    """
    if len(volumes) < PERIOD_VOLUME_SMA:
        return np.nan
    
    return np.mean(volumes[-PERIOD_VOLUME_SMA:])


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Detects Bollinger Band rejection signals with MFI and volume confirmation.
    A Buy signal is generated on a bullish rejection from the lower band with oversold MFI turning up and high volume.
    A Sell signal is generated on a bearish rejection from the upper band with overbought MFI turning down and high volume.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm
        
        # Ensure sufficient data for all indicator calculations
        if len(candles) < MIN_CANDLES:
            continue

        # Extract relevant data as numpy arrays for efficient calculations
        close_prices = np.array([c.close for c in candles])
        high_prices = np.array([c.high for c in candles])
        low_prices = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        # --- Calculate Indicators ---
        _, upper_bb, lower_bb = _calculate_bollinger_bands(close_prices)
        current_mfi, previous_mfi = _calculate_mfi(candles)
        volume_sma = _calculate_volume_sma(volumes)

        # Skip if any indicator calculation resulted in NaN due to insufficient data
        if np.isnan(upper_bb) or np.isnan(lower_bb) or \
           np.isnan(current_mfi) or np.isnan(previous_mfi) or np.isnan(volume_sma):
            continue

        # --- Get Current Candle Data ---
        last_candle = candles[-1]
        current_close = last_candle.close
        current_high = last_candle.high
        current_low = last_candle.low
        current_volume = last_candle.volume
        
        # --- Buy Signal Condition: Bullish Rejection ---
        # 1. Price dipped below but closed above Lower BB
        bb_rejection_buy = (current_low < lower_bb and current_close > lower_bb)
        
        # 2. MFI is oversold AND turning upward
        mfi_confirmation_buy = (current_mfi <= MFI_OVERSOLD_THRESHOLD and current_mfi > previous_mfi)
        
        # 3. High volume confirmation (current volume is significantly higher than average)
        volume_confirmation_buy = (current_volume > (volume_sma * VOLUME_THRESHOLD_MULTIPLIER))
        
        if bb_rejection_buy and mfi_confirmation_buy and volume_confirmation_buy:
            signals.append(BuySignal(
                pair=pair,
                timestamp=last_candle.hour,
                price=current_close,
                rule_id="fc565741-5374-43c3-bb88-15bfc512405b",
                confidence=0.9 
            ))

        # --- Sell Signal Condition: Bearish Rejection ---
        # 1. Price rose above but closed below Upper BB
        bb_rejection_sell = (current_high > upper_bb and current_close < upper_bb)
        
        # 2. MFI is overbought AND turning downward
        mfi_confirmation_sell = (current_mfi >= MFI_OVERBOUGHT_THRESHOLD and current_mfi < previous_mfi)
        
        # 3. High volume confirmation (current volume is significantly higher than average)
        volume_confirmation_sell = (current_volume > (volume_sma * VOLUME_THRESHOLD_MULTIPLIER))
        
        if bb_rejection_sell and mfi_confirmation_sell and volume_confirmation_sell:
            signals.append(SellSignal(
                pair=pair,
                timestamp=last_candle.hour,
                price=current_close,
                rule_id="fc565741-5374-43c3-bb88-15bfc512405b",
                confidence=0.9
            ))

    return signals