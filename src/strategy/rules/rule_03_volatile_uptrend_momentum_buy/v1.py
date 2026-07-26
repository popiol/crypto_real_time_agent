from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

# Constants for indicator periods and thresholds
RSI_PERIOD = 14
BB_PERIOD = 20
BBW_THRESHOLD = 5.0  # %
KC_EMA_PERIOD = 20
KC_ATR_PERIOD = 10
KC_ATR_MULTIPLIER = 2
KCP_THRESHOLD = 120.0  # %
MIN_CANDLES = 20  # Minimum warm candles required for all calculations

def _calculate_rsi(close_prices: np.ndarray, period: int) -> float | None:
    """
    Calculates the Relative Strength Index (RSI) for the last price in the series.
    Uses Wilder's smoothing method.
    """
    if len(close_prices) < period + 1:
        return None

    # Calculate price differences
    deltas = np.diff(close_prices)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Initial average gain and loss (SMA for the first 'period' values)
    # These averages are calculated over the first 'period' differences
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Apply Wilder's smoothing for subsequent periods if more data exists
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    # Handle division by zero for RS calculation
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0 # If no losses, RSI is 100. If no gains and no losses, RSI is 50.

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calculate_sma(prices: np.ndarray, period: int) -> float | None:
    """
    Calculates the Simple Moving Average (SMA) of the last 'period' prices.
    """
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def _calculate_std_dev(prices: np.ndarray, period: int) -> float | None:
    """
    Calculates the Standard Deviation of the last 'period' prices.
    """
    if len(prices) < period:
        return None
    return np.std(prices[-period:])

def _calculate_ema(prices: np.ndarray, period: int) -> float | None:
    """
    Calculates the Exponential Moving Average (EMA) for the last price in the series.
    """
    if len(prices) < period:
        return None
    
    # Calculate initial SMA for the first 'period' values
    ema = np.mean(prices[:period])
    
    multiplier = 2 / (period + 1)
    # Apply EMA formula for subsequent prices
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def _calculate_atr(candles: list, period: int) -> float | None:
    """
    Calculates the Average True Range (ATR) based on a simple moving average of True Ranges.
    """
    if len(candles) < period + 1: # Need at least period+1 candles to calculate 'period' true ranges
        return None

    true_ranges = []
    # Calculate True Ranges starting from the second candle
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i-1].close
        
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    
    # Return the simple moving average of the last 'period' true ranges
    if len(true_ranges) < period: # This check handles edge cases, though unlikely if initial check passes
        return None
    
    return np.mean(true_ranges[-period:])

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the "Volatile Uptrend Momentum Buy" rule.
    Triggers a buy signal when RSI > 65, BBW > 5%, and Keltner Channel Percentage > 120%.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_tick = pair_data.hot[-1] if pair_data.hot else None

        # Ensure we have enough warm candles and a current tick
        if not hot_tick or len(warm_candles) < MIN_CANDLES:
            continue

        close_prices = np.array([candle.close for candle in warm_candles])
        # high_prices = np.array([candle.high for candle in warm_candles]) # Not directly used as numpy array for ATR
        # low_prices = np.array([candle.low for candle in warm_candles]) # Not directly used as numpy array for ATR

        # 1. Calculate RSI
        rsi_value = _calculate_rsi(close_prices, RSI_PERIOD)
        if rsi_value is None:
            continue

        # 2. Calculate Bollinger Bands Width
        sma_bb = _calculate_sma(close_prices, BB_PERIOD)
        std_dev_bb = _calculate_std_dev(close_prices, BB_PERIOD)
        if sma_bb is None or std_dev_bb is None:
            continue
        
        upper_bb = sma_bb + (2 * std_dev_bb)
        lower_bb = sma_bb - (2 * std_dev_bb)
        
        # Avoid division by zero if SMA is zero (unlikely for price data)
        if sma_bb == 0:
            continue
        bbw_value = ((upper_bb - lower_bb) / sma_bb) * 100

        # 3. Calculate Keltner Channel Percentage
        ema_kc = _calculate_ema(close_prices, KC_EMA_PERIOD)
        atr_kc = _calculate_atr(warm_candles, KC_ATR_PERIOD)
        if ema_kc is None or atr_kc is None:
            continue
        
        upper_kc = ema_kc + (atr_kc * KC_ATR_MULTIPLIER)
        lower_kc = ema_kc - (atr_kc * KC_ATR_MULTIPLIER)

        # Avoid division by zero if Keltner Channel width is zero
        kc_width = upper_kc - lower_kc
        if kc_width == 0:
            continue
        
        keltner_channel_percentage_value = ((hot_tick.last_price - lower_kc) / kc_width) * 100

        # Check all conditions for a BUY signal
        if (rsi_value > 65 and
                bbw_value > BBW_THRESHOLD and
                keltner_channel_percentage_value > KCP_THRESHOLD):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=hot_tick.polled_at,
                price=hot_tick.last_price,
                rule_id="volatile_uptrend_momentum_buy",
                indicators={
                    "rsi": rsi_value,
                    "bbw": bbw_value,
                    "kcp": keltner_channel_percentage_value,
                }
            ))

    return signals