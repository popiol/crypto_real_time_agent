from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

# Rule ID for this specific trading strategy
RULE_ID = "a20f92ce-5bb1-45ae-8518-ece5b01787f9"

# Parameters as defined in the pseudocode
SHORT_EMA_PERIOD = 10
MEDIUM_EMA_PERIOD = 20  # Adjusted from 30
LONG_EMA_PERIOD = 50    # Adjusted from 100
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 40  # Adjusted from 50
RSI_SELL_THRESHOLD = 60 # Adjusted from 50
VOLUME_AVG_PERIOD = 20

# --- Helper Functions for Indicator Calculations ---

def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates the Exponential Moving Average (EMA) for a given price series."""
    if len(prices) < period:
        return np.array([])
    
    ema = np.zeros_like(prices, dtype=float)
    # Initialize the first EMA value with a Simple Moving Average (SMA)
    ema[period - 1] = np.mean(prices[:period])
    alpha = 2 / (period + 1)
    
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    
    return ema

def calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates the Relative Strength Index (RSI) for a given price series."""
    # Need at least period + 1 prices to calculate the first 'period' differences
    if len(prices) <= period:
        return np.array([])

    diff = np.diff(prices)
    gains = np.where(diff > 0, diff, 0)
    losses = np.where(diff < 0, -diff, 0) # Make losses positive for calculation

    avg_gain = np.zeros_like(gains, dtype=float)
    avg_loss = np.zeros_like(losses, dtype=float)

    # Calculate initial average gain/loss over the period using SMA
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    # Apply Wilder's smoothing method for subsequent averages
    for i in range(period, len(gains)):
        avg_gain[i] = ((avg_gain[i-1] * (period - 1)) + gains[i]) / period
        avg_loss[i] = ((avg_loss[i-1] * (period - 1)) + losses[i]) / period
    
    # Calculate Relative Strength (RS) and RSI
    # Handle division by zero for avg_loss by setting RS to infinity where avg_loss is zero
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.inf), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Return only the valid RSI values, which start from the point where the first average is computable
    return rsi[period - 1:]

# --- Main Signal Generation Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Identifies strong trend initiation/continuation using triple EMA crossover,
    RSI confirmation, and volume above average, with adjusted thresholds.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm
        ticks = pair_data.hot

        # --- Data Sufficiency Checks ---
        # Minimum candles needed for the longest EMA (LONG_EMA_PERIOD) and RSI (RSI_PERIOD + 1)
        min_candles_required = max(LONG_EMA_PERIOD, RSI_PERIOD + 1) 
        if len(candles) < min_candles_required:
            # Not enough historical candle data to calculate all indicators
            continue

        # Minimum ticks needed for volume average (VOLUME_AVG_PERIOD) and current volume
        # We need at least VOLUME_AVG_PERIOD ticks to calculate the average,
        # and one more for the current volume, making it VOLUME_AVG_PERIOD + 1.
        min_ticks_required = VOLUME_AVG_PERIOD + 1 
        if len(ticks) < min_ticks_required:
            # Not enough recent tick data for volume analysis
            continue

        # --- Prepare Data for Indicators ---
        close_prices = np.array([c.close for c in candles], dtype=float)
        # Use volume_24h from ticks. This is an approximation as it's not candle-specific volume,
        # but the only volume data available for recent periods.
        tick_volumes_24h = np.array([t.volume_24h for t in ticks], dtype=float)
        
        # Get the latest data points for signal evaluation
        current_candle_close_price = close_prices[-1]
        current_candle_timestamp = candles[-1].hour # Timestamp of the latest candle close
        latest_tick_price = ticks[-1].last_price    # Most recent actual trade price
        current_volume_24h = tick_volumes_24h[-1]    # Most recent 24h volume from tick

        # --- Calculate Indicators ---
        ema_short = calculate_ema(close_prices, SHORT_EMA_PERIOD)
        ema_medium = calculate_ema(close_prices, MEDIUM_EMA_PERIOD)
        ema_long = calculate_ema(close_prices, LONG_EMA_PERIOD)
        rsi_value = calculate_rsi(close_prices, RSI_PERIOD)

        # Ensure all indicator arrays have enough values for current and previous checks
        # For EMA crossover, we need at least 2 values (current and previous).
        # For EMA_long and RSI, we only need the latest value (1).
        if len(ema_short) < 2 or len(ema_medium) < 2 or len(ema_long) < 1 or len(rsi_value) < 1:
            continue
        
        # Calculate volume average over the last VOLUME_AVG_PERIOD ticks
        # np.mean(tick_volumes_24h[-VOLUME_AVG_PERIOD-1:-1]) would be the average of the *previous* VOLUME_AVG_PERIOD ticks,
        # excluding the current one. The pseudocode implies the average of the last `VOLUME_AVG_PERIOD` including current or up to current.
        # Given `volume > avg_volume`, it implies the current volume is compared against a past average.
        # So, we take the average of the `VOLUME_AVG_PERIOD` ticks *before* the current one.
        volume_avg = np.mean(tick_volumes_24h[-VOLUME_AVG_PERIOD-1:-1])


        # --- Extract Current and Previous Indicator Values ---
        ema_short_current = ema_short[-1]
        ema_short_previous = ema_short[-2]
        ema_medium_current = ema_medium[-1]
        ema_medium_previous = ema_medium[-2]
        
        ema_long_current = ema_long[-1]
        rsi_current = rsi_value[-1]

        # --- Buy Signal Conditions ---
        # 1. Short EMA crosses above Medium EMA
        buy_crossover = (ema_short_current > ema_medium_current) and \
                        (ema_short_previous <= ema_medium_previous)
        # 2. Price is above Long EMA
        buy_price_above_long_ema = current_candle_close_price > ema_long_current
        # 3. RSI confirms bullish momentum (adjusted threshold)
        buy_rsi_confirm = rsi_current > RSI_BUY_THRESHOLD
        # 4. Volume confirms conviction (adjusted condition: above average)
        buy_volume_confirm = current_volume_24h > volume_avg

        if buy_crossover and buy_price_above_long_ema and buy_rsi_confirm and buy_volume_confirm:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle_timestamp, # Use candle close timestamp
                price=latest_tick_price,            # Use latest tick price for execution
                rule_id=RULE_ID
            ))

        # --- Sell Signal Conditions ---
        # 1. Short EMA crosses below Medium EMA
        sell_crossover = (ema_short_current < ema_medium_current) and \
                         (ema_short_previous >= ema_medium_previous)
        # 2. Price is below Long EMA
        sell_price_below_long_ema = current_candle_close_price < ema_long_current
        # 3. RSI confirms bearish momentum (adjusted threshold)
        sell_rsi_confirm = rsi_current < RSI_SELL_THRESHOLD
        # 4. Volume confirms conviction (adjusted condition: above average)
        sell_volume_confirm = current_volume_24h > volume_avg

        if sell_crossover and sell_price_below_long_ema and sell_rsi_confirm and sell_volume_confirm:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle_timestamp, # Use candle close timestamp
                price=latest_tick_price,            # Use latest tick price for execution
                rule_id=RULE_ID
            ))

    return signals