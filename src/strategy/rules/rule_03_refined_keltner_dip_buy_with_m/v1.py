from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, WarmCandle, SellSignal
from datetime import datetime

# Constants for indicator periods and multipliers
KELTNER_PERIOD = 20
RSI_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV_MULTIPLIER = 2.0

# Minimum data requirements
# Keltner/RSI need N+1 candles for True Range / price changes calculation
# Bollinger needs N candles for SMA/StdDev
MIN_WARM_CANDLES_REQUIRED = max(KELTNER_PERIOD + 1, RSI_PERIOD + 1, BOLLINGER_PERIOD)
MIN_HOT_TICKS_REQUIRED = 1

def calculate_ema(prices: list[float], period: int) -> float | None:
    """
    Calculates the Exponential Moving Average (EMA) for the last price in the list.
    Returns None if insufficient data.
    """
    if len(prices) < period:
        return None
    
    # Use SMA for the initial EMA value
    ema = np.mean(prices[:period])
    
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices)):
        ema = (prices[i] - ema) * multiplier + ema
    return ema

def calculate_atr(candles: list[WarmCandle], period: int) -> float | None:
    """
    Calculates the Average True Range (ATR) for the last candle.
    Returns None if insufficient data.
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
    
    if len(true_ranges) < period:
        return None
        
    # ATR is typically a Simple Moving Average of True Ranges
    return np.mean(true_ranges[-period:])

def calculate_rsi(closes: list[float], period: int) -> float | None:
    """
    Calculates the Relative Strength Index (RSI) for the last close price.
    Uses Wilder's smoothing method. Returns None if insufficient data.
    """
    if len(closes) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(change if change > 0 else 0)
        losses.append(abs(change) if change < 0 else 0)

    # Initial average gain/loss (SMA) over the first 'period' changes
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Calculate first RSI value
    if avg_loss == 0:
        rs = np.inf if avg_gain > 0 else 0
    else:
        rs = avg_gain / avg_loss
    
    rsi = 100 - (100 / (1 + rs)) if rs != np.inf else 100

    # Wilder's smoothing for subsequent values
    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

        if avg_loss == 0:
            rs = np.inf if avg_gain > 0 else 0
        else:
            rs = avg_gain / avg_loss
        
        rsi = 100 - (100 / (1 + rs)) if rs != np.inf else 100
    
    return rsi

def calculate_bb_width(closes: list[float], period: int, std_dev_multiplier: float) -> float | None:
    """
    Calculates the Bollinger Band Width (BBW) for the last close price.
    Returns None if insufficient data.
    """
    if len(closes) < period:
        return None
    
    recent_closes = np.array(closes[-period:])
    
    sma = np.mean(recent_closes)
    std_dev = np.std(recent_closes)
    
    if sma == 0: # Avoid division by zero, though unlikely with price data
        return None
        
    # BBW = (Upper Band - Lower Band) / Middle Band
    # Upper Band = SMA + K * StdDev, Lower Band = SMA - K * StdDev
    # So, BBW = (2 * K * StdDev) / SMA
    bb_width = (2 * std_dev_multiplier * std_dev) / sma
    return bb_width


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates a buy signal based on Keltner Channel Percentage, RSI, Bollinger Band Width,
    and Tick Bid-Ask Spread Percentage.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # 1. Insufficient data check
        if len(warm_candles) < MIN_WARM_CANDLES_REQUIRED or len(hot_ticks) < MIN_HOT_TICKS_REQUIRED:
            continue

        # Extract close prices for indicator calculations
        closes = [c.close for c in warm_candles]
        
        current_close = closes[-1]

        # 2. Calculate Keltner Channel Percentage
        # Keltner Middle Band is EMA of closes
        keltner_ema = calculate_ema(closes, KELTNER_PERIOD)
        keltner_atr = calculate_atr(warm_candles, KELTNER_PERIOD)

        if keltner_ema is None or keltner_atr is None or keltner_atr == 0:
            continue
        
        # KCP = (Close - Middle Band) / ATR * 100
        keltner_channel_percentage = (current_close - keltner_ema) / keltner_atr * 100

        # 3. Calculate RSI
        rsi = calculate_rsi(closes, RSI_PERIOD)
        if rsi is None:
            continue

        # 4. Calculate Bollinger Band Width
        bb_width = calculate_bb_width(closes, BOLLINGER_PERIOD, BOLLINGER_STD_DEV_MULTIPLIER)
        if bb_width is None:
            continue

        # 5. Get current Tick Bid-Ask Spread Percentage
        tick_bid_ask_spread_percentage = hot_ticks[-1].spread_rel

        # Apply all conditions for a BUY signal
        cond_kcp = -50 <= keltner_channel_percentage <= 20
        cond_rsi = 35 <= rsi <= 55
        cond_bbw = bb_width < 0.08
        cond_spread = tick_bid_ask_spread_percentage < 0.5

        if cond_kcp and cond_rsi and cond_bbw and cond_spread:
            signals.append(BuySignal(
                pair=pair,
                timestamp=hot_ticks[-1].polled_at,
                price=hot_ticks[-1].last_price,
                rule_id="keltner_dip_buy_volatility_filter_v1",
                indicators={
                    "keltner_channel_percentage": keltner_channel_percentage,
                    "rsi": rsi,
                    "bollinger_band_width": bb_width,
                    "tick_bid_ask_spread_percentage": tick_bid_ask_spread_percentage,
                }
            ))
    return signals