from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime
from pydantic import BaseModel, Field # Needed for type hints in helper functions if not already imported

# Helper functions for indicators (copied from existing implementation)
def calculate_ema(closes: list[float], period: int) -> list[float]:
    """Calculates the Exponential Moving Average (EMA)."""
    if len(closes) < period:
        # Not enough data to calculate EMA for the given period
        return []

    np_closes = np.array(closes)
    ema = np.zeros_like(np_closes)
    
    # Initialize the first EMA value with the Simple Moving Average (SMA) of the first 'period' closes
    ema[period - 1] = np.mean(np_closes[:period])

    alpha = 2 / (period + 1)
    
    # Calculate EMA for the remaining values
    for i in range(period, len(np_closes)):
        ema[i] = (np_closes[i] - ema[i-1]) * alpha + ema[i-1]
        
    return ema.tolist()

def calculate_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    """Calculates the Average True Range (ATR) using Wilder's smoothing method."""
    if len(highs) < period or len(lows) < period or len(closes) < period:
        # Not enough data to calculate ATR for the given period
        return []

    np_highs = np.array(highs)
    np_lows = np.array(lows)
    np_closes = np.array(closes)

    # Ensure all arrays are of the same length, trimming from the start if necessary
    min_len = min(len(np_highs), len(np_lows), len(np_closes))
    np_highs = np_highs[-min_len:]
    np_lows = np_lows[-min_len:]
    np_closes = np_closes[-min_len:]
    
    # Calculate previous closes, handling the first element (no previous close)
    prev_closes = np.roll(np_closes, 1)
    prev_closes[0] = np_closes[0] # Using current close as previous for the first element is a common approximation

    # Calculate True Range (TR)
    # TR = max(H - L, abs(H - C_prev), abs(L - C_prev))
    tr = np.maximum(np_highs - np_lows, np.abs(np_highs - prev_closes), np.abs(np_lows - prev_closes))

    atr = np.zeros(min_len)
    
    # Initial ATR is the Simple Moving Average (SMA) of the first 'period' TR values
    if min_len < period: # This check should ideally be redundant due to initial check, but good for robustness
        return []
        
    atr[period - 1] = np.mean(tr[:period])

    # Wilder's smoothing factor for ATR: alpha = 1 / period
    alpha = 1 / period

    # Calculate ATR for the remaining values
    for i in range(period, min_len):
        atr[i] = (tr[i] * alpha) + (atr[i-1] * (1 - alpha))
        
    return atr.tolist()

def calculate_rsi(closes: list[float], period: int) -> list[float]:
    """Calculates the Relative Strength Index (RSI)."""
    # Need at least period + 1 closes to calculate 'period' differences
    if len(closes) < period + 1:
        return []

    np_closes = np.array(closes)
    
    # Calculate price changes (deltas)
    deltas = np.diff(np_closes) # deltas will have len(closes) - 1 elements
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, abs(deltas), 0)

    avg_gain = np.zeros_like(gains)
    avg_loss = np.zeros_like(losses)

    # Initial average gain/loss (SMA for the first 'period' values of gains/losses)
    # The first valid index for avg_gain/loss will be period - 1
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    # Smoothed moving average for subsequent values (Wilder's smoothing)
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

    rs = np.zeros_like(avg_gain)
    rsi = np.zeros_like(avg_gain)

    # Calculate RS and RSI, avoiding division by zero for avg_loss
    # RSI values are valid from index 'period - 1' onwards
    for i in range(period - 1, len(avg_gain)):
        if avg_loss[i] == 0:
            rs[i] = np.inf # If no losses, RS is infinite
            rsi[i] = 100.0 # RSI is 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
            
    # Return only the valid RSI values (from index period-1 to the end)
    return rsi[period-1:].tolist()


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    
    for pair, pair_data in data.items():
        # Minimum data checks as per pseudocode and indicator requirements
        # Keltner (EMA 20, ATR 10) requires at least 20 warm candles.
        # RSI 14 requires at least 14 + 1 = 15 warm candles.
        # EMA 5 requires at least 5 warm candles.
        # So, 20 warm candles is the overall minimum for warm data.
        if len(pair_data.warm) < 20:
            continue 
        
        # We need at least one hot tick for current price, spread.
        if len(pair_data.hot) < 1:
            continue

        # Extract warm data for indicators
        closes = [candle.close for candle in pair_data.warm]
        highs = [candle.high for candle in pair_data.warm]
        lows = [candle.low for candle in pair_data.warm]
        current_warm_close = closes[-1]

        # Extract hot data for current market conditions
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        
        # --- Calculate Relative Strength Index (RSI) (14 periods) ---
        rsi_14_series = calculate_rsi(closes, 14)
        if not rsi_14_series: # Should be covered by initial len check, but for robustness
            continue
        rsi_14 = rsi_14_series[-1]

        # --- Calculate Keltner Channel Percentage (20-period EMA for midline, 10-period ATR for band) ---
        ema_20_series = calculate_ema(closes, 20)
        atr_10_series = calculate_atr(highs, lows, closes, 10) # Pseudocode specifies ATR 10
        
        if not ema_20_series or not atr_10_series: # Should be covered by initial len check
            continue

        keltner_midline = ema_20_series[-1]
        atr_value = atr_10_series[-1] # ATR for the Keltner band calculation
        
        keltner_channel_percentage: float
        # The pseudocode implies using data.warm[-1].close for Keltner calculation, not current_price from hot data.
        # Avoid division by zero and ensure ATR is positive for a meaningful channel.
        # The Keltner channel width is typically a multiple of ATR, e.g., 2*ATR.
        if atr_value * 2 <= 0:
            keltner_channel_percentage = 0.0 # Treat as neutral if ATR is zero or negative (highly unlikely)
        else:
            keltner_channel_percentage = ((current_warm_close - keltner_midline) / (atr_value * 2)) * 100

        # --- Calculate 5-period Exponential Moving Average (EMA) for confirmation ---
        ema_5_series = calculate_ema(closes, 5)
        if not ema_5_series: # Should be covered by initial len check
            continue
        ema_5 = ema_5_series[-1]

        # --- Get Bid-Ask Spread Percentage from hot data ---
        # The Tick model has 'spread_rel' which is (ask - bid) / mid * 100 (%).
        # The rule states 'spread_percentage < 0.005', implying 0.005 percentage points.
        bid_ask_spread_percentage = current_tick.spread_rel

        # --- Apply trading conditions for a BUY signal ---
        # 1. Sufficient warm data: len(pair_data.warm) >= 20 (already checked)
        # 2. RSI 14 is oversold but not extremely so: 30 <= RSI <= 40
        # 3. Keltner Channel Percentage indicates price is within or slightly above/below channel: -10 <= Keltner % <= 10
        # 4. Current hourly close price is above its 5-period EMA: data.warm[-1].close > EMA(5)
        # 5. Bid-ask spread is very tight: data.hot[-1].spread_rel < 0.005
        
        if (rsi_14 >= 30 and 
            rsi_14 <= 40 and 
            keltner_channel_percentage >= -10 and 
            keltner_channel_percentage <= 10 and 
            current_warm_close > ema_5 and 
            bid_ask_spread_percentage < 0.005):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_tick.polled_at, # Use hot data timestamp for signal
                price=current_price, # Use hot data last_price for signal
                rule_id="oversold_reversal_ema_confirm",
                reason="Enhanced Mean Reversion Entry: RSI and Keltner in defined oversold range, EMA confirmation, and tight spread."
            ))

    return signals