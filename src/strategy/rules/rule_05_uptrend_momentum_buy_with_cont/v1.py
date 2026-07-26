from __future__ import annotations
import numpy as np
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from datetime import datetime

# --- Helper functions for indicators ---

def calculate_rsi(candles: list[WarmCandle], period: int = 14) -> float | None:
    """
    Calculates the Relative Strength Index (RSI) for the last candle.
    Requires `period + 1` candles for calculation (period differences).
    """
    if len(candles) < period + 1:
        return None

    closes = np.array([c.close for c in candles])
    
    # Calculate price differences (deltas)
    # np.diff(closes) will yield len(closes) - 1 differences.
    # If len(closes) is 15 (for period=14), diffs will have 14 elements.
    diffs = np.diff(closes)

    # Separate gains and losses
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)

    avg_gain = np.zeros_like(gains)
    avg_loss = np.zeros_like(losses)

    # Initial averages (simple average for the first 'period' diffs)
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    # Wilders smoothing (RMA) for subsequent averages
    for i in range(period, len(diffs)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    # Use the last calculated average gain/loss
    final_avg_gain = avg_gain[-1]
    final_avg_loss = avg_loss[-1]

    if final_avg_loss == 0:
        return 100.0 if final_avg_gain > 0 else 50.0 # No losses, infinite RSI or neutral
    
    rs = final_avg_gain / final_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(data: np.ndarray, period: int) -> float | None:
    """Calculates Simple Moving Average (SMA) of the last 'period' values."""
    if len(data) < period:
        return None
    return np.mean(data[-period:])

def calculate_atr(candles: list[WarmCandle], period: int = 10) -> float | None:
    """
    Calculates Average True Range (ATR) for the last candle.
    Requires `period + 1` candles to calculate `period` True Ranges.
    """
    if len(candles) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i-1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    # At this point, len(true_ranges) is len(candles) - 1.
    # If len(candles) is period + 1, then len(true_ranges) is period.
    if len(true_ranges) < period: # Should not happen if initial check is correct
        return None

    atr_values = np.zeros(len(true_ranges))
    
    # Initial ATR is SMA of the first 'period' True Ranges
    atr_values[period - 1] = np.mean(true_ranges[:period])

    # Wilders smoothing for subsequent ATRs
    for i in range(period, len(true_ranges)):
        atr_values[i] = (atr_values[i-1] * (period - 1) + true_ranges[i]) / period
    
    return atr_values[-1] # Return the last ATR value

def calculate_keltner_channel_percentage(candles: list[WarmCandle], sma_period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> float | None:
    """
    Calculates Keltner Channel Percentage (KCP) for the last candle.
    KCP = ((Close - Lower Band) / (Upper Band - Lower Band)) * 100.
    Requires enough candles for both SMA (sma_period) and ATR (atr_period + 1).
    """
    # Minimum candles needed: max of SMA period and ATR period + 1 for ATR calculation
    if len(candles) < max(sma_period, atr_period + 1):
        return None

    closes = np.array([c.close for c in candles])
    
    # Calculate Middle Line (SMA of closes)
    middle_line = calculate_sma(closes, sma_period)
    if middle_line is None:
        return None
    
    # Calculate ATR
    current_atr = calculate_atr(candles, atr_period)
    if current_atr is None:
        return None
    
    # Upper/Lower Bands (using a standard multiplier of 2.0)
    upper_band = middle_line + (current_atr * multiplier)
    lower_band = middle_line - (current_atr * multiplier)

    current_close = closes[-1]

    # If UB == LB (e.g., ATR is 0), avoid division by zero.
    if upper_band == lower_band:
        return 100.0 # Price is on the middle line
    
    kcp = ((current_close - lower_band) / (upper_band - lower_band)) * 100
    return kcp

def calculate_bollinger_bands_width(candles: list[WarmCandle], period: int = 20, std_dev_multiplier: float = 2.0) -> float | None:
    """
    Calculates Bollinger Bands Width (BBW) for the last candle.
    BBW = ((Upper Band - Lower Band) / Middle Band) * 100.
    Requires `period` candles for SMA and standard deviation.
    """
    if len(candles) < period:
        return None

    closes = np.array([c.close for c in candles])

    # Calculate Middle Band (SMA of closes)
    middle_band = calculate_sma(closes, period)
    if middle_band is None:
        return None

    # Calculate Standard Deviation of the last 'period' closes
    std_dev = np.std(closes[-period:])
    
    # Upper/Lower Bands (using a standard multiplier of 2.0)
    upper_band = middle_band + (std_dev * std_dev_multiplier)
    lower_band = middle_band - (std_dev * std_dev_multiplier)

    # If Middle Band is zero (unlikely for price data), avoid division by zero.
    if middle_band == 0:
        return 0.0
        
    bbw = ((upper_band - lower_band) / middle_band) * 100
    return bbw

# --- Main signal function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Minimum warm data check
        # RSI(14) needs 15 candles (period + 1).
        # KCP (SMA=20, ATR=10) needs 20 candles (max(20, 10+1)).
        # BBW(20) needs 20 candles (period).
        # The longest lookback is 20 candles.
        if len(pair_data.warm) < 20:
            continue

        warm_candles = pair_data.warm
        latest_tick = pair_data.hot[-1] if pair_data.hot else None

        # Need the latest tick for spread calculation
        if latest_tick is None:
            continue

        # Calculate Indicators for the most recent hourly candle (warm_candles[-1])
        current_rsi = calculate_rsi(warm_candles, period=14)
        current_kcp = calculate_keltner_channel_percentage(warm_candles, sma_period=20, atr_period=10)
        current_bbw = calculate_bollinger_bands_width(warm_candles, period=20)
        
        # Bid-Ask Spread Percentage calculation
        # Ensure last_price is not zero to avoid division by zero
        if latest_tick.last_price == 0:
            current_bid_ask_spread_pct = None
        else:
            current_bid_ask_spread_pct = (latest_tick.ask_price - latest_tick.bid_price) / latest_tick.last_price * 100

        # If any indicator calculation failed (returned None), skip this pair
        if any(x is None for x in [current_rsi, current_kcp, current_bbw, current_bid_ask_spread_pct]):
            continue

        # Define Conditions based on the pseudocode
        CONDITION_RSI = (current_rsi > 65 and current_rsi < 75)
        CONDITION_KCP = (current_kcp > 130 and current_kcp < 180)
        CONDITION_BBW = (current_bbw > 5 and current_bbw < 15)
        CONDITION_SPREAD = (current_bid_ask_spread_pct > 0.5 and current_bid_ask_spread_pct < 3.5)

        # Generate Buy Signal if all conditions are met
        if CONDITION_RSI and CONDITION_KCP and CONDITION_BBW and CONDITION_SPREAD:
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_tick.polled_at,
                price=latest_tick.last_price,
            ))
            
    return signals