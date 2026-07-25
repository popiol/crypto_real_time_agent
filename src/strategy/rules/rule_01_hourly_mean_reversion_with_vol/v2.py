from __future__ import annotations
import math
from src.agent.models import BuySignal, MarketData, SellSignal

# Rule ID as per the idea description
RULE_ID = "keltner-atr-multiple-increase-v1"

# Constants from pseudocode
N_KC = 20  # Keltner Channel period (hourly) for EMA
N_RSI = 14  # RSI period (hourly)
N_ATR = 20  # ATR period for Keltner Channel (hourly)
VOLUME_ANOMALY_PERIOD = 20  # Volume Anomaly period (hourly)
NEW_ATR_MULTIPLE = 2.0  # Proposed new multiple for Keltner Channel

# --- Helper Functions for Indicator Calculations ---

def calculate_ema(prices: list[float], period: int) -> float:
    """
    Calculates the Exponential Moving Average (EMA) for the last value in the series.
    Requires at least `period` prices.
    """
    if len(prices) < period:
        return float('nan')

    # Initialize EMA with a simple moving average of the first 'period' values
    ema = sum(prices[:period]) / period
    alpha = 2 / (period + 1)

    # Apply the EMA formula for the rest of the series
    for i in range(period, len(prices)):
        ema = prices[i] * alpha + ema * (1 - alpha)
    return ema

def calculate_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
    """
    Calculates the Average True Range (ATR).
    Requires at least `period + 1` candles to calculate `period` True Ranges.
    """
    if len(closes) < period + 1:
        return float('nan')

    true_ranges = []
    # Calculate True Ranges for each candle from the second one
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i-1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    # Average the last 'period' True Ranges
    if len(true_ranges) < period: # This check should ideally be covered by the initial len(closes) check
        return float('nan')

    return sum(true_ranges[-period:]) / period

def calculate_rsi(closes: list[float], period: int) -> float:
    """
    Calculates the Relative Strength Index (RSI).
    Requires at least `period + 1` closes to calculate `period` price changes.
    """
    if len(closes) < period + 1:
        return float('nan')

    gains = []
    losses = []

    # Calculate price changes, gains, and losses
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(0.0, change))
        losses.append(abs(min(0.0, change)))

    if len(gains) < period: # Should be caught by the initial len(closes) check
        return float('nan')

    # Calculate initial average gain and loss (Simple Moving Average for the first 'period' values)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Apply Wilder's smoothing method for subsequent values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        # If there are no losses, RSI is 100 if there were any gains, otherwise 50 (if no net change)
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_volume_anomaly(volumes: list[float], period: int) -> float:
    """
    Calculates Volume Anomaly: current_volume / SMA(past_volumes, period).
    Requires at least `period + 1` volumes (1 for current, `period` for past average).
    """
    if len(volumes) < period + 1:
        return float('nan')

    current_volume = volumes[-1]
    
    # Calculate SMA of volumes for the 'period' candles *before* the current one
    # This means taking volumes from index `len(volumes) - period - 1` up to `len(volumes) - 2`
    past_volumes = volumes[-(period + 1):-1]
    
    # Ensure we actually got 'period' past volumes
    if len(past_volumes) != period: # Should be covered by the initial len(volumes) check
        return float('nan')

    avg_past_volume = sum(past_volumes) / period
    
    if avg_past_volume == 0:
        return float('inf') if current_volume > 0 else 0.0 # Handle division by zero
    
    return current_volume / avg_past_volume

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # 1. Data Sufficiency Checks
        # Minimum candles needed for each indicator:
        # EMA(N_KC): N_KC candles
        # ATR(N_ATR): N_ATR + 1 candles
        # RSI(N_RSI): N_RSI + 1 candles
        # Volume Anomaly(VOLUME_ANOMALY_PERIOD): VOLUME_ANOMALY_PERIOD + 1 candles
        
        required_warm_candles = max(
            N_KC,
            N_ATR + 1,
            N_RSI + 1,
            VOLUME_ANOMALY_PERIOD + 1
        )
        # For current constants: max(20, 21, 15, 21) = 21
        
        if len(warm_candles) < required_warm_candles:
            continue

        if not hot_ticks:
            continue

        # Extract relevant data lists from warm candles
        closes = [candle.close for candle in warm_candles]
        highs = [candle.high for candle in warm_candles]
        lows = [candle.low for candle in warm_candles]
        volumes = [candle.volume for candle in warm_candles]

        current_tick = hot_ticks[-1]
        current_price = current_tick.last_price
        polled_at = current_tick.polled_at

        # 2. Calculate Indicators
        ema_last = calculate_ema(closes, N_KC)
        atr_last = calculate_atr(highs, lows, closes, N_ATR)
        rsi_last = calculate_rsi(closes, N_RSI)
        volume_anomaly_last = calculate_volume_anomaly(volumes, VOLUME_ANOMALY_PERIOD)

        # Check for NaN results from indicator calculations (insufficient data or division by zero cases)
        if any(math.isnan(val) for val in [ema_last, atr_last, rsi_last, volume_anomaly_last]):
            continue

        # 3. Calculate Keltner Channel Position
        keltner_center = ema_last
        keltner_channel_deviation = NEW_ATR_MULTIPLE * atr_last
        
        if keltner_channel_deviation == 0: # Avoid division by zero if ATR is zero (unlikely but possible)
            continue 
            
        keltner_position = (current_price - keltner_center) / keltner_channel_deviation

        # 4. Apply Signal Conditions (BUY only)
        # Keltner_Position >= -0.1 AND Keltner_Position <= 0.4
        # RSI_N.last() >= 30 AND RSI_N.last() <= 45
        # Volume_Anomaly.last() > 1.0
        
        if (keltner_position >= -0.1 and keltner_position <= 0.4) and \
           (rsi_last >= 30 and rsi_last <= 45) and \
           (volume_anomaly_last > 1.0):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=polled_at,
                price=current_price,
                rule_id=RULE_ID,
            ))

    return signals