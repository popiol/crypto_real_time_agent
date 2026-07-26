from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle
from datetime import datetime

# --- Helper Functions for Indicators ---

def _calculate_sma(data: np.ndarray, period: int) -> float | None:
    """Calculates the Simple Moving Average for the last 'period' elements of the data."""
    if len(data) < period:
        return None
    return np.mean(data[-period:])

def _calculate_stddev(data: np.ndarray, period: int) -> float | None:
    """Calculates the Standard Deviation for the last 'period' elements of the data."""
    if len(data) < period:
        return None
    return np.std(data[-period:])

def _calculate_ema(data: np.ndarray, period: int) -> float | None:
    """Calculates the Exponential Moving Average for the given data."""
    if len(data) < period:
        return None
    
    # Calculate SMA for the initial EMA value
    ema = np.mean(data[:period])
    multiplier = 2 / (period + 1)
    
    # Calculate EMA for the rest of the data
    for price in data[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def _calculate_atr(warm_candles: list[WarmCandle], period: int) -> float | None:
    """Calculates the Average True Range (ATR) based on Wilder's smoothing method."""
    # Need at least period + 1 candles to calculate ATR (for previous close)
    if len(warm_candles) < period + 1:
        return None

    true_ranges = []
    # Start from the second candle to get previous close
    for i in range(1, len(warm_candles)):
        current_candle = warm_candles[i]
        prev_close = warm_candles[i-1].close
        
        tr1 = current_candle.high - current_candle.low
        tr2 = abs(current_candle.high - prev_close)
        tr3 = abs(current_candle.low - prev_close)
        true_ranges.append(max(tr1, tr2, tr3))
    
    if len(true_ranges) < period:
        return None

    # Calculate initial ATR as SMA of the first 'period' true ranges
    atr = np.mean(true_ranges[:period])
    
    # Apply Wilder's smoothing for subsequent ATR values
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period

    return atr

def calculate_rsi(closes: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI)."""
    # Need at least period + 1 closes to calculate period gains/losses and the first average
    if len(closes) < period + 1:
        return None

    closes_arr = np.array(closes, dtype=float)
    deltas = np.diff(closes_arr)

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate initial average gain/loss over the first 'period' deltas
    # These arrays will have len(closes)-1 elements.
    # We need the last 'period' deltas to calculate the current RSI.
    # So, we take the last `period` elements of `gains` and `losses` to calculate the current RSI.
    # However, for correct smoothing, we need to calculate the EMA-like average over the entire relevant history.

    # If we only need the *last* RSI value, we can calculate smoothed averages iteratively.
    
    # Initial average gain/loss
    avg_gain = np.mean(gains[-(period):])
    avg_loss = np.mean(losses[-(period):])

    # If the last 'period' values are used for initial calculation, this is a simplified approach.
    # For a full EMA-like smoothing, we'd need to compute averages over the entire `gains` and `losses` history.
    # Given `data.warm` limit and focus on the latest candle, we need the RSI for the latest candle.
    # The standard RSI calculation involves an initial SMA, then subsequent EMA-like smoothing.

    # To calculate RSI for the *last* candle, we need `period` previous values for the averages.
    # So, we need `period` values for gains/losses, which means `period+1` closes.
    # The `gains` and `losses` arrays are `len(closes) - 1` long.
    # To get the RSI for `closes[-1]`, we need `gains[-period:]` and `losses[-period:]` for the first average.
    # Or, to be precise with smoothing:
    
    # Calculate initial average gain/loss for the first `period` deltas
    initial_avg_gain = np.mean(gains[:period])
    initial_avg_loss = np.mean(losses[:period])

    # Apply smoothing
    smoothed_avg_gain = initial_avg_gain
    smoothed_avg_loss = initial_avg_loss
    
    for i in range(period, len(gains)):
        smoothed_avg_gain = (smoothed_avg_gain * (period - 1) + gains[i]) / period
        smoothed_avg_loss = (smoothed_avg_loss * (period - 1) + losses[i]) / period

    if smoothed_avg_loss == 0:
        return 100.0 if smoothed_avg_gain > 0 else 50.0 # If no losses, RSI is 100. If no gains either, 50.
    
    rs = smoothed_avg_gain / smoothed_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_keltner_channel_percentage(warm_candles: list[WarmCandle], ema_period: int, atr_period: int, multiplier: float) -> float | None:
    """Calculates Keltner Channel Percentage (KCP).
    KCP is interpreted as 100 + ((Close - Midline) / (ATR * Multiplier)) * 100, where 100 is midline.
    """
    # Need enough data for both EMA and ATR calculations
    if len(warm_candles) < max(ema_period, atr_period + 1): # ATR needs period + 1
        return None
    
    closes = np.array([c.close for c in warm_candles], dtype=float)
    
    # Midline (EMA of closes)
    midline = _calculate_ema(closes, ema_period)
    if midline is None:
        return None

    # ATR
    atr_value = _calculate_atr(warm_candles, atr_period)
    if atr_value is None or atr_value == 0: # Avoid division by zero
        return None

    last_close = closes[-1]
    
    # KCP calculation based on the rule's implied interpretation
    # 100 represents the midline. Values > 100 mean price is above midline.
    # KCP = 100 + ((Close - Midline) / (ATR * Multiplier)) * 100
    kcp = 100 + ((last_close - midline) / (atr_value * multiplier)) * 100
    return kcp

def calculate_bollinger_bands_width(closes: list[float], sma_period: int, stddev_multiplier: float) -> float | None:
    """Calculates Bollinger Bands Width (BBW)."""
    if len(closes) < sma_period:
        return None
    
    closes_arr = np.array(closes, dtype=float)
    
    # Middle Band (SMA)
    middle_band = _calculate_sma(closes_arr, sma_period)
    if middle_band is None:
        return None
    
    # Standard Deviation
    std_dev = _calculate_stddev(closes_arr, sma_period)
    if std_dev is None:
        return None

    # Upper and Lower Bands
    upper_band = middle_band + (std_dev * stddev_multiplier)
    lower_band = middle_band - (std_dev * stddev_multiplier)

    # Bollinger Bands Width calculation
    # BBW = ((Upper Band - Lower Band) / Middle Band) * 100
    if middle_band == 0: # Avoid division by zero
        return None
        
    bbw = ((upper_band - lower_band) / middle_band) * 100
    return bbw

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Uptrend Continuation Buy with Moderate Momentum and Liquidity' rule.
    Triggers a BUY signal when:
    - 14-period RSI is between 65 and 75 (moderate bullish momentum).
    - Keltner Channel Percentage (20-period EMA, 20-period ATR, multiplier 2.0) is between 135 and 165
      (healthy but not extreme price extension).
    - 20-period Bollinger Bands Width is between 5 and 15 (moderate volatility).
    - Current Bid-Ask Spread Percentage is >= 0.5% (sufficient liquidity).
    """
    signals: list[BuySignal | SellSignal] = []
    
    for pair, pair_data in data.items():
        # Ensure sufficient warm data for 20-period indicators.
        # ATR (period=20) calculation requires 'period + 1' candles. So, 21 candles minimum.
        if len(pair_data.warm) < 21:
            continue
        
        # Ensure hot data is available for current price and spread
        if not pair_data.hot:
            continue
        
        # Extract necessary data for calculations
        warm_closes = [c.close for c in pair_data.warm]
        
        # Get current bid/ask for real-time spread from the latest tick
        current_tick = pair_data.hot[-1]
        current_bid = current_tick.bid_price
        current_ask = current_tick.ask_price
        timestamp = current_tick.polled_at

        # Calculate Bid-Ask Spread Percentage
        # Ensure current_bid is not zero to prevent division by zero
        spread_pct = ((current_ask - current_bid) / current_bid) * 100 if current_bid > 0 else 0.0

        # Calculate indicators
        rsi_value = calculate_rsi(warm_closes, period=14)
        kcp_value = calculate_keltner_channel_percentage(pair_data.warm, ema_period=20, atr_period=20, multiplier=2.0)
        bbw_value = calculate_bollinger_bands_width(warm_closes, sma_period=20, stddev_multiplier=2.0)
        
        # Check if all indicators were successfully calculated (not None)
        if any(v is None for v in [rsi_value, kcp_value, bbw_value]):
            continue

        # Entry conditions for BUY/LONG
        if (65 <= rsi_value <= 75 and
            135 <= kcp_value <= 165 and
            5 <= bbw_value <= 15 and
            spread_pct >= 0.5):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_ask, # Buy at the ask price
                rule_id="moderate_uptrend_continuation_buy",
                indicators={
                    "rsi_14": rsi_value,
                    "kcp_20_20_2.0": kcp_value,
                    "bbw_20_2.0": bbw_value,
                    "spread_pct": spread_pct,
                }
            ))
            
    return signals