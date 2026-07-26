from __future__ import annotations
import statistics
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from datetime import datetime

# --- Helper Functions for Indicators ---

def _calculate_ema(data: list[float], period: int) -> float:
    """Calculates the Exponential Moving Average (EMA)."""
    if len(data) < period:
        return np.nan
    
    # Calculate initial SMA for the first 'period' values
    sma = sum(data[:period]) / period
    ema_values = [sma]
    
    alpha = 2 / (period + 1)
    
    for i in range(period, len(data)):
        ema = (data[i] * alpha) + (ema_values[-1] * (1 - alpha))
        ema_values.append(ema)
        
    return ema_values[-1]

def _calculate_true_ranges(candles: list[WarmCandle]) -> list[float]:
    """Calculates True Range for each candle.
    TR_i = max(high_i - low_i, abs(high_i - close_i-1), abs(low_i - close_i-1))
    For the first candle, TR is simply high - low as there's no previous close.
    """
    true_ranges = []
    if not candles:
        return true_ranges

    # For the first candle, use High - Low
    true_ranges.append(candles[0].high - candles[0].low)
    
    for i in range(1, len(candles)):
        current_candle = candles[i]
        prev_close = candles[i-1].close
        
        tr1 = current_candle.high - current_candle.low
        tr2 = abs(current_candle.high - prev_close)
        tr3 = abs(current_candle.low - prev_close)
        
        true_ranges.append(max(tr1, tr2, tr3))
        
    return true_ranges

def _calculate_atr(candles: list[WarmCandle], period: int) -> float:
    """Calculates the Average True Range (ATR) using EMA smoothing."""
    true_ranges = _calculate_true_ranges(candles)
    
    if len(true_ranges) < period:
        return np.nan

    # Initial SMA of True Ranges for the first 'period' values
    atr_values = [sum(true_ranges[:period]) / period]
    
    alpha = 2 / (period + 1)
    
    for i in range(period, len(true_ranges)):
        atr = (true_ranges[i] * alpha) + (atr_values[-1] * (1 - alpha))
        atr_values.append(atr)
        
    return atr_values[-1]

def _calculate_rsi(candles: list[WarmCandle], period: int) -> float:
    """Calculates the Relative Strength Index (RSI)."""
    # Need period + 1 candles to get 'period' price changes for initial SMA
    if len(candles) < period + 1: 
        return np.nan

    gains = []
    losses = []

    for i in range(1, len(candles)):
        change = candles[i].close - candles[i-1].close
        gains.append(change if change > 0 else 0)
        losses.append(abs(change) if change < 0 else 0)

    # After loop, len(gains) == len(candles) - 1. We need at least 'period' changes.
    if len(gains) < period: 
         return np.nan

    # Initial SMA for average gain/loss
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # EMA smoothing
    alpha = 2 / (period + 1)

    for i in range(period, len(gains)):
        avg_gain = (gains[i] * alpha) + (avg_gain * (1 - alpha))
        avg_loss = (losses[i] * alpha) + (avg_loss * (1 - alpha))

    if avg_loss == 0:
        # If no losses, RSI is 100 (if there are gains) or 50 (if no gains/losses)
        return 100.0 if avg_gain > 0 else 50.0 
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calculate_keltner_channel_percentage(
    candles: list[WarmCandle], 
    ema_period: int, 
    atr_period: int, 
    atr_multiplier: float
) -> float:
    """Calculates the Keltner Channel Percentage.
    KCP = ((latest_close - Middle_Line) / (ATR * Multiplier)) * 100
    """
    # Ensure enough data for both EMA and ATR components
    # EMA needs 'ema_period' candles. ATR needs 'atr_period' candles.
    if len(candles) < max(ema_period, atr_period): 
        return np.nan

    closes = [c.close for c in candles]
    
    middle_line = _calculate_ema(closes, ema_period)
    if np.isnan(middle_line):
        return np.nan
    
    atr = _calculate_atr(candles, atr_period)
    # ATR can be 0 if prices are completely flat for the entire period,
    # which would lead to division by zero.
    if np.isnan(atr) or atr == 0: 
        return np.nan
    
    latest_close = candles[-1].close
    
    kcp = ((latest_close - middle_line) / (atr * atr_multiplier)) * 100
    
    return kcp

# --- Main Signal Function ---

# Rule parameters
KC_EMA_PERIOD = 20
KC_ATR_PERIOD = 10 # A common ATR period used with Keltner Channels
KC_ATR_MULTIPLIER = 2.0 # Standard deviation multiplier for Keltner Channels
RSI_PERIOD = 14

# Minimum candles required for all indicators:
# EMA(20) needs 20 candles.
# ATR(10) needs 10 candles.
# RSI(14) needs 14 + 1 = 15 candles.
MIN_CANDLES_REQUIRED = max(KC_EMA_PERIOD, KC_ATR_PERIOD, RSI_PERIOD + 1) # This evaluates to 20

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # 1. Insufficient data check for warm candles and hot ticks
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue
        if not hot_ticks: # Need at least one tick for current price/spread
            continue

        latest_tick = hot_ticks[-1]

        # 2. Calculate Keltner Channel Percentage
        keltner_channel_percentage = _calculate_keltner_channel_percentage(
            warm_candles, KC_EMA_PERIOD, KC_ATR_PERIOD, KC_ATR_MULTIPLIER
        )
        if np.isnan(keltner_channel_percentage):
            continue

        # 3. Calculate RSI
        rsi = _calculate_rsi(warm_candles, RSI_PERIOD)
        if np.isnan(rsi):
            continue

        # 4. Calculate Bid-Ask Spread Percentage
        # Ensure bid_price, ask_price, and last_price are valid and non-zero
        if latest_tick.bid_price <= 0 or latest_tick.ask_price <= 0 or latest_tick.last_price <= 0:
            continue
        
        bid_ask_spread_percentage = (latest_tick.ask_price - latest_tick.bid_price) / latest_tick.last_price * 100

        # 5. Apply Trading Rule Conditions for a BUY signal
        if (
            keltner_channel_percentage < -50
            and rsi > 55
            and bid_ask_spread_percentage < 2.0
        ):
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=latest_tick.polled_at,
                    price=latest_tick.last_price,
                    rule_id="keltner_rsi_dip_buy_v1",
                    indicators={
                        "keltner_channel_percentage": keltner_channel_percentage,
                        "rsi": rsi,
                        "bid_ask_spread_percentage": bid_ask_spread_percentage,
                    }
                )
            )
    return signals