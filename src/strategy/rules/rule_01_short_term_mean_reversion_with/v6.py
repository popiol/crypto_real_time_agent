from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

# Helper functions for indicators
def calculate_ema(closes: list[float], period: int) -> list[float]:
    """Calculates the Exponential Moving Average (EMA)."""
    if len(closes) < period:
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
    min_len = min(len(highs), len(lows), len(closes))
    if min_len < period:
        return []

    np_highs = np.array(highs[-min_len:])
    np_lows = np.array(lows[-min_len:])
    np_closes = np.array(closes[-min_len:])
    
    # Calculate previous closes, handling the first element (no previous close)
    prev_closes = np.roll(np_closes, 1)
    prev_closes[0] = np_closes[0] # Using current close as previous for the first element is a common approximation

    # Calculate True Range (TR)
    # TR = max(H - L, abs(H - C_prev), abs(L - C_prev))
    tr = np.maximum(np_highs - np_lows, np.abs(np_highs - prev_closes), np.abs(np_lows - prev_closes))

    atr = np.zeros(min_len)
    
    # Initial ATR is the Simple Moving Average (SMA) of the first 'period' TR values
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
    if len(gains) < period:
        return [] # Should not happen if len(closes) >= period + 1

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

def calculate_sma(data: list[float], period: int) -> list[float]:
    """Calculates the Simple Moving Average (SMA)."""
    if len(data) < period:
        return []
    np_data = np.array(data)
    # Using np.convolve for efficiency, 'valid' mode returns only full overlaps
    sma = np.convolve(np_data, np.ones(period)/period, mode='valid')
    return sma.tolist()

def calculate_stddev(data: list[float], period: int) -> list[float]:
    """Calculates the Rolling Standard Deviation."""
    if len(data) < period:
        return []
    np_data = np.array(data)
    # Calculate rolling standard deviation. The output length will be len(data) - period + 1
    stddevs = [np.std(np_data[i-period+1 : i+1]) for i in range(period-1, len(np_data))]
    return stddevs

def calculate_bollinger_band_width_percentage(closes: list[float], period: int = 20, num_std_dev: float = 2.0) -> list[float]:
    """
    Calculates the Bollinger Band Width as a percentage of the middle band.
    BBW = ((Upper Band - Lower Band) / Middle Band) * 100
    """
    if len(closes) < period:
        return []

    sma_series = calculate_sma(closes, period)
    stddev_series = calculate_stddev(closes, period)

    # If SMA or StdDev cannot be calculated, return empty
    if not sma_series or not stddev_series:
        return []

    bb_width_percentages = []
    # sma_series and stddev_series align: sma_series[i] and stddev_series[i] correspond
    # to the window ending at closes[period-1+i].
    for i in range(len(sma_series)):
        mid_band = sma_series[i]
        std_dev = stddev_series[i]
        
        # (Upper Band - Lower Band) = (mid_band + num_std_dev * std_dev) - (mid_band - num_std_dev * std_dev)
        # = 2 * num_std_dev * std_dev
        
        # Calculate width as percentage of midline
        if mid_band > 0:
            bb_width = (2 * num_std_dev * std_dev / mid_band) * 100
            bb_width_percentages.append(bb_width)
        else:
            # If mid_band is zero or negative (unlikely for prices), treat width as 0
            bb_width_percentages.append(0.0) 

    return bb_width_percentages

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    
    # Define indicator periods and parameters
    RSI_PERIOD = 14
    EMA_KELTNER_PERIOD = 20
    ATR_KELTNER_PERIOD = 10
    KELTNER_ATR_MULTIPLIER = 2.0 # Common multiplier for Keltner Channel bands
    BB_PERIOD = 20
    BB_NUM_STD_DEV = 2.0 # Standard multiplier for Bollinger Bands

    # Determine the maximum lookback period needed for warm data
    # RSI: 14 + 1 = 15
    # EMA for Keltner: 20
    # ATR for Keltner: 10
    # BB: 20
    MIN_WARM_CANDLES = max(RSI_PERIOD + 1, EMA_KELTNER_PERIOD, ATR_KELTNER_PERIOD, BB_PERIOD)
    
    for pair, pair_data in data.items():
        # Minimum data checks
        if len(pair_data.warm) < MIN_WARM_CANDLES:
            continue 
        
        if len(pair_data.hot) < 1:
            continue

        # Extract warm data for indicators
        closes = [candle.close for candle in pair_data.warm]
        highs = [candle.high for candle in pair_data.warm]
        lows = [candle.low for candle in pair_data.warm]

        # Extract hot data for current market conditions
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        
        # --- Calculate Relative Strength Index (RSI) ---
        rsi_series = calculate_rsi(closes, RSI_PERIOD)
        if not rsi_series:
            continue
        rsi_value = rsi_series[-1]

        # --- Calculate Keltner Channel Percentage ---
        ema_keltner_series = calculate_ema(closes, EMA_KELTNER_PERIOD)
        atr_keltner_series = calculate_atr(highs, lows, closes, ATR_KELTNER_PERIOD)
        
        if not ema_keltner_series or not atr_keltner_series:
            continue

        keltner_midline = ema_keltner_series[-1]
        atr_value = atr_keltner_series[-1]
        
        keltner_channel_percentage: float
        channel_width = atr_value * KELTNER_ATR_MULTIPLIER
        if channel_width <= 0: # Avoid division by zero or negative width
            keltner_channel_percentage = 0.0 
        else:
            keltner_channel_percentage = ((current_price - keltner_midline) / channel_width) * 100

        # --- Calculate Bollinger Band Width Percentage ---
        bb_width_series = calculate_bollinger_band_width_percentage(closes, BB_PERIOD, BB_NUM_STD_DEV)
        if not bb_width_series:
            continue
        bollinger_band_width = bb_width_series[-1]

        # --- Get Bid-Ask Spread Percentage from hot data ---
        # `spread_rel` in Tick model is already a percentage (e.g., 0.01 for 0.01%)
        bid_ask_spread_percentage = current_tick.spread_rel

        # --- Apply trading conditions for a BUY signal ---
        # Pseudocode: IF Hourly_RSI(14) < 35 AND Hourly_KCP(20) < -20 AND Hourly_BBW(20) > 1.25 AND Bid_Ask_Spread_Percentage_Last_Tick < 0.001 THEN GENERATE BUY SIGNAL
        
        # Condition 1: Hourly RSI < 35
        cond_rsi = rsi_value < 35
        
        # Condition 2: Hourly Keltner Channel Percentage < -20
        cond_keltner = keltner_channel_percentage < -20
        
        # Condition 3: Hourly Bollinger Band Width > 1.25 (percentage, e.g., 1.25 means 1.25%)
        cond_bb_width = bollinger_band_width > 1.25
        
        # Condition 4: Bid-Ask Spread Percentage < 0.001 (relative spread, e.g., 0.001 means 0.001%)
        cond_spread = bid_ask_spread_percentage < 0.001

        if cond_rsi and cond_keltner and cond_bb_width and cond_spread:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_tick.polled_at,
                price=current_price,
                rule_id="oversold-mean-reversion-v2", 
                reason=f"Loosened Oversold Mean Reversion: RSI < 35 ({rsi_value:.2f}), KCP < -20 ({keltner_channel_percentage:.2f}%), BB Width > 1.25 ({bollinger_band_width:.2f}%), Spread < 0.001% ({bid_ask_spread_percentage:.4f}%)",
                indicators={
                    "rsi_value": rsi_value,
                    "keltner_channel_percentage": keltner_channel_percentage,
                    "bollinger_band_width_percentage": bollinger_band_width,
                    "bid_ask_spread_percentage": bid_ask_spread_percentage
                }
            ))

    return signals