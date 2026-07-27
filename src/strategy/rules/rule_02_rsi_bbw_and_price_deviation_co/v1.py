from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Constants ---
SMA_PERIOD = 12
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD_DEV = 2

# Minimum number of candles required for all indicators
# RSI needs at least RSI_PERIOD + 1 candles for its initial calculation.
MIN_CANDLES_REQUIRED = max(SMA_PERIOD, RSI_PERIOD + 1, BB_PERIOD)

# --- Helper Functions for Indicators ---

def calculate_sma(closes: np.ndarray, period: int) -> float | None:
    """Calculates the Simple Moving Average."""
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))

def calculate_rsi(closes: np.ndarray, period: int) -> float | None:
    """Calculates the Relative Strength Index."""
    if len(closes) < period + 1: # Need period + 1 for initial gain/loss calculation
        return None

    # Calculate price changes (deltas)
    deltas = np.diff(closes)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate initial average gain and loss over the period
    # We take the last 'period' deltas to calculate the RSI for the most recent candle
    avg_gain = np.mean(gains[-(period):])
    avg_loss = np.mean(losses[-(period):])

    # Handle edge cases for RS calculation
    if avg_loss == 0:
        return 100.0 # If no losses, RSI is 100
    if avg_gain == 0:
        return 0.0   # If no gains, RSI is 0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)

def calculate_bollinger_bands(closes: np.ndarray, period: int, std_dev: int) -> tuple[float | None, float | None, float | None]:
    """Calculates Bollinger Bands (Upper, Middle, Lower)."""
    if len(closes) < period:
        return None, None, None

    # Calculate SMA for the middle band
    middle_band = np.mean(closes[-period:])
    
    # Calculate standard deviation
    std = np.std(closes[-period:])
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return float(upper_band), float(middle_band), float(lower_band)

def calculate_bbw(closes: np.ndarray, period: int, std_dev: int) -> float | None:
    """Calculates Bollinger Band Width."""
    upper_band, middle_band, lower_band = calculate_bollinger_bands(closes, period, std_dev)
    
    if upper_band is None or middle_band is None or lower_band is None:
        return None
    
    # Avoid division by zero if middle_band is zero (highly unlikely with price data)
    if middle_band == 0:
        return None
        
    return (upper_band - lower_band) / middle_band

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        hot_ticks = pair_data.hot
        warm_candles = pair_data.warm

        # Ensure we have at least one hot tick for the current price and timestamp
        if not hot_ticks:
            continue
        
        # Ensure sufficient warm candle data for all indicator calculations
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        last_price = hot_ticks[-1].last_price
        timestamp = hot_ticks[-1].polled_at

        # Extract close prices from warm candles for indicator calculations
        closes = np.array([c.close for c in warm_candles])

        # --- Calculate Indicators ---
        
        sma = calculate_sma(closes, SMA_PERIOD)
        if sma is None:
            continue # Should not happen if MIN_CANDLES_REQUIRED is met

        rsi = calculate_rsi(closes, RSI_PERIOD)
        if rsi is None:
            continue # Should not happen if MIN_CANDLES_REQUIRED is met

        bbw = calculate_bbw(closes, BB_PERIOD, BB_STD_DEV)
        if bbw is None:
            continue # Should not happen if MIN_CANDLES_REQUIRED is met

        # Price Deviation from SMA
        if sma == 0: # Avoid division by zero, though highly unlikely with real price data
            continue
        price_deviation_from_sma = (last_price - sma) / sma

        # --- Buy Conditions ---
        buy_condition_1 = (price_deviation_from_sma > 0.0 and price_deviation_from_sma < 0.05)
        buy_condition_2 = (rsi >= 50.0 and rsi <= 65.0)
        buy_condition_3 = (bbw >= 0.02 and bbw <= 0.06)

        if buy_condition_1 and buy_condition_2 and buy_condition_3:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=last_price,
                rule_id="rsi_bbw_pdev_trend_conf"
            ))

        # --- Sell Conditions ---
        sell_condition_1 = (price_deviation_from_sma < 0.0)
        sell_condition_2 = (rsi < 45.0)
        sell_condition_3 = (rsi > 60.0)
        sell_condition_4 = (price_deviation_from_sma > 0.05)

        if sell_condition_1 or sell_condition_2 or sell_condition_3 or sell_condition_4:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=last_price,
                rule_id="rsi_bbw_pdev_trend_conf"
            ))

    return signals