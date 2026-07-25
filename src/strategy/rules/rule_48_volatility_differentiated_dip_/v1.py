"""Rule VolDiffDipRecov_001 — Volatility-Differentiated Dip Recovery Buy."""
from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal

# Rule constants
RULE_ID = "VolDiffDipRecov_001"
BB_PERIOD = 20  # Period for Bollinger Bands and SMA on warm candles
BB_STD_DEV_MULTIPLIER = 2  # Standard deviation multiplier for Bollinger Bands

# Data sufficiency requirements
MIN_WARM_CANDLES = BB_PERIOD
MIN_HOT_TICKS = 1

# Rule specific thresholds
LOW_VOL_BB_WIDTH_THRESHOLD = 0.2
HIGH_VOL_BB_WIDTH_TARGET = 1.0
LOW_VOL_MRO_THRESHOLD = -0.5
HIGH_VOL_MRO_THRESHOLD = -2.0
HOT_SPREAD_TO_DEPTH_RATIO_THRESHOLD = 0.0001

# Epsilon for floating point comparison, especially for HIGH_VOL_BB_WIDTH_TARGET
FLOAT_COMPARISON_EPSILON = 1e-6

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # --- 1. Check hot data for SpreadToDepthRatio ---
        hot_ticks = pair_data.hot
        if len(hot_ticks) < MIN_HOT_TICKS:
            continue

        last_tick = hot_ticks[-1]
        
        # Calculate HotSpreadToDepthRatio
        spread_abs = last_tick.spread_abs
        total_depth_volume = last_tick.bid_volume + last_tick.ask_volume

        if total_depth_volume <= 0:  # Avoid division by zero or non-positive volume
            continue
        
        hot_spread_to_depth_ratio = spread_abs / total_depth_volume

        # --- 2. Check warm data for Bollinger Band Width and Mean Reversion Oscillator ---
        warm_candles = pair_data.warm
        if len(warm_candles) < MIN_WARM_CANDLES:
            continue

        # Use numpy for efficient calculations on candle close prices
        # Ensure we have enough candles for the BB_PERIOD
        close_prices = np.array([c.close for c in warm_candles])
        recent_close_prices = close_prices[-BB_PERIOD:]

        if len(recent_close_prices) < BB_PERIOD:
            # This case should ideally be caught by len(warm_candles) < MIN_WARM_CANDLES
            # but is included for defensive programming.
            continue 

        sma = np.mean(recent_close_prices)
        if sma <= 0:  # Avoid division by zero or non-positive SMA
            continue

        std_dev = np.std(recent_close_prices)

        # Calculate WarmBollingerBandWidth (relative width)
        # BBW = (Upper Band - Lower Band) / Middle Band = (2 * K * StdDev) / SMA
        warm_bb_width = (2 * BB_STD_DEV_MULTIPLIER * std_dev) / sma

        # Calculate WarmMeanReversionOscillator
        # MRO = (Current Price - SMA) / SMA
        current_close = warm_candles[-1].close
        warm_mro = (current_close - sma) / sma

        # --- 3. Apply the trading rule conditions ---
        
        # Condition for low-volatility dip
        condition_low_vol = (warm_bb_width < LOW_VOL_BB_WIDTH_THRESHOLD) and \
                            (warm_mro < LOW_VOL_MRO_THRESHOLD)
        
        # Condition for high-volatility dip
        # Use epsilon for float comparison with HIGH_VOL_BB_WIDTH_TARGET
        condition_high_vol = (abs(warm_bb_width - HIGH_VOL_BB_WIDTH_TARGET) < FLOAT_COMPARISON_EPSILON) and \
                             (warm_mro < HIGH_VOL_MRO_THRESHOLD)
        
        # Combined rule logic: either volatility condition met AND HotSpreadToDepthRatio is low
        if (condition_low_vol or condition_high_vol) and \
           (hot_spread_to_depth_ratio < HOT_SPREAD_TO_DEPTH_RATIO_THRESHOLD):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=last_tick.polled_at,
                price=last_tick.last_price,
                rule_id=RULE_ID,
                confidence=1.0  # Default confidence as no specific calculation is given
            ))

    return signals