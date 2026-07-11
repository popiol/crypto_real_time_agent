from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Parameters ---
EMA_PERIOD = 20
ATR_PERIOD = 10
ATR_MULTIPLIER = 2.0
STOCH_K_PERIOD = 14
# STOCH_D_PERIOD = 3 # Not used in signal conditions, so not implemented
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80

# Unique identifier for this trading rule
RULE_ID = "c488fe2f-3e81-464d-b810-c291a11abb65"

# Minimum number of candles required for calculations.
# EMA needs EMA_PERIOD candles.
# ATR needs ATR_PERIOD true ranges, which requires ATR_PERIOD + 1 candles (for previous close).
# Stochastic %K needs STOCH_K_PERIOD candles.
MIN_CANDLES = max(EMA_PERIOD, ATR_PERIOD + 1, STOCH_K_PERIOD)

# --- Helper Functions for Indicator Calculations ---

def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Exponential Moving Average (EMA)."""
    if len(data) < period:
        return np.array([])
    
    ema_values = np.zeros_like(data, dtype=float)
    alpha = 2 / (period + 1)
    
    # Initialize the first EMA value with a Simple Moving Average (SMA)
    ema_values[period - 1] = np.mean(data[:period])
    
    # Calculate subsequent EMA values
    for i in range(period, len(data)):
        ema_values[i] = (data[i] * alpha) + (ema_values[i-1] * (1 - alpha))
    
    return ema_values

def _calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Calculates Average True Range (ATR)."""
    # Need N+1 candles to calculate N True Range values (due to previous close)
    if len(highs) < period + 1: 
        return np.array([])

    true_ranges = np.zeros(len(highs) - 1, dtype=float)
    for i in range(1, len(highs)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        true_ranges[i-1] = max(tr1, tr2, tr3)

    if len(true_ranges) < period:
        return np.array([])

    atr_values = np.zeros_like(true_ranges, dtype=float)
    
    # Initialize the first ATR value with a Simple Moving Average of True Ranges
    atr_values[period - 1] = np.mean(true_ranges[:period])
    
    # Calculate subsequent ATR values using the standard smoothing formula
    for i in range(period, len(true_ranges)):
        atr_values[i] = (atr_values[i-1] * (period - 1) + true_ranges[i]) / period
        
    return atr_values

def _calculate_stochastic_k(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int) -> np.ndarray:
    """Calculates Stochastic Oscillator %K."""
    if len(closes) < period:
        return np.array([])

    percent_k = np.zeros_like(closes, dtype=float)
    
    for i in range(period - 1, len(closes)):
        # Find the lowest low and highest high over the lookback period
        period_low = np.min(lows[i - period + 1 : i + 1])
        period_high = np.max(highs[i - period + 1 : i + 1])
        
        diff = period_high - period_low
        if diff != 0:
            percent_k[i] = ((closes[i] - period_low) / diff) * 100
        else:
            # Handle division by zero: if high == low over the period,
            # %K is typically 50 or the previous value. Using 50 for simplicity
            # if this is the first valid point, otherwise previous value.
            percent_k[i] = percent_k[i-1] if i > period - 1 else 50.0
            
    return percent_k


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Keltner Channel Mean Reversion with Stochastic Oscillator Confirmation.

    This rule detects mean-reversion opportunities when the price deviates significantly
    from its exponential moving average (EMA) as defined by Keltner Channels, and
    confirms the reversal potential with the Stochastic Oscillator.

    A Buy signal is generated when the price closes below the lower Keltner Channel
    and the Stochastic Oscillator (%K) is below a predefined oversold threshold (e.g., 20).

    A Sell signal is generated when the price closes above the upper Keltner Channel
    and the Stochastic Oscillator (%K) is above a predefined overbought threshold (e.g., 80).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm
        
        # Ensure sufficient data for indicator calculations
        if len(candles) < MIN_CANDLES:
            continue

        # Extract required candle data into numpy arrays for efficient calculation
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        
        # --- Calculate Keltner Channels ---
        mid_band_values = _calculate_ema(closes, EMA_PERIOD)
        if mid_band_values.size == 0: continue # Not enough data for EMA

        atr_values = _calculate_atr(highs, lows, closes, ATR_PERIOD)
        if atr_values.size == 0: continue # Not enough data for ATR

        # Get the latest indicator values
        current_mid_band = mid_band_values[-1]
        current_atr = atr_values[-1]

        upper_band = current_mid_band + (current_atr * ATR_MULTIPLIER)
        lower_band = current_mid_band - (current_atr * ATR_MULTIPLIER)

        # --- Calculate Stochastic Oscillator ---
        percent_k_values = _calculate_stochastic_k(closes, highs, lows, STOCH_K_PERIOD)
        if percent_k_values.size == 0: continue # Not enough data for Stochastic %K
        
        current_percent_k = percent_k_values[-1]

        # --- Generate Signals based on the latest candle ---
        current_close = closes[-1]
        current_timestamp = candles[-1].hour
        
        # Buy Signal condition:
        # Price closes below the lower Keltner Channel AND Stochastic %K is oversold
        if current_close < lower_band and current_percent_k < STOCH_OVERSOLD:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id=RULE_ID
            ))
        # Sell Signal condition:
        # Price closes above the upper Keltner Channel AND Stochastic %K is overbought
        elif current_close > upper_band and current_percent_k > STOCH_OVERBOUGHT:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id=RULE_ID
            ))

    return signals