from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, Tick, WarmCandle

# Rule-specific constants
RULE_ID = "MR_ModVol_TVI_Spread_Refined"
BB_LOOKBACK_PERIOD = 20  # Fits data.warm (max 24 candles)

# Thresholds as defined in the rule idea
MODERATE_BB_WIDTH_MIN = 0.02
MODERATE_BB_WIDTH_MAX = 0.08
SIGNIFICANT_TVI_BUY_THRESHOLD = -1000
SIGNIFICANT_TVI_SELL_THRESHOLD = 1000
MAX_SPREAD_PERCENTAGE = 0.01  # 1.0%

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # --- Data Acquisition & Validation ---
        if not pair_data.hot:
            continue
        
        latest_tick: Tick = pair_data.hot[-1]
        polled_at = latest_tick.polled_at

        # Validate latest_tick prices to prevent division by zero or invalid calculations
        if latest_tick.bid_price <= 0 or latest_tick.ask_price <= 0:
            continue

        # Calculate Tick Bid-Ask Spread Percentage from the latest tick
        # As per pseudocode: (ask - bid) / bid
        tick_spread_percentage = (latest_tick.ask_price - latest_tick.bid_price) / latest_tick.bid_price

        # Calculate Tick Volume Imbalance from data.hot
        tick_volume_imbalance = 0
        for tick in pair_data.hot:
            # As per pseudocode: sum of (ask_volume - bid_volume)
            tick_volume_imbalance += tick.ask_volume - tick.bid_volume
        
        # Hourly Indicator Calculation (from data.warm)
        if len(pair_data.warm) < BB_LOOKBACK_PERIOD:
            continue
        
        # Get the last BB_LOOKBACK_PERIOD closing prices
        hourly_closes = np.array([c.close for c in pair_data.warm[-BB_LOOKBACK_PERIOD:]])

        # Calculate SMA and Standard Deviation
        hourly_sma = np.mean(hourly_closes)
        # Using population standard deviation (ddof=0) for Bollinger Bands
        hourly_std_dev = np.std(hourly_closes, ddof=0)

        # Handle cases where SMA might be zero or negative
        if hourly_sma <= 0:
            continue

        # Calculate Bollinger Bands Width Percentage
        # Assuming 2-sigma bands as per common BBands definition
        hourly_upper_band = hourly_sma + (2 * hourly_std_dev)
        hourly_lower_band = hourly_sma - (2 * hourly_std_dev)
            
        bb_width_percentage = (hourly_upper_band - hourly_lower_band) / hourly_sma

        # --- Rule Logic ---
        # Check all conditions for a Buy Signal
        if (MODERATE_BB_WIDTH_MIN < bb_width_percentage < MODERATE_BB_WIDTH_MAX and
            tick_volume_imbalance < SIGNIFICANT_TVI_BUY_THRESHOLD and
            tick_spread_percentage < MAX_SPREAD_PERCENTAGE):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=polled_at,
                price=latest_tick.last_price, # As per pseudocode
                rule_id=RULE_ID,
                indicators={
                    "bb_width_percentage": bb_width_percentage,
                    "tick_volume_imbalance": float(tick_volume_imbalance),
                    "tick_spread_percentage": tick_spread_percentage,
                    "hourly_sma": hourly_sma,
                    "hourly_std_dev": hourly_std_dev
                },
                confidence=1.0, # Default confidence
            ))

        # Check all conditions for a Sell Signal
        elif (MODERATE_BB_WIDTH_MIN < bb_width_percentage < MODERATE_BB_WIDTH_MAX and
              tick_volume_imbalance > SIGNIFICANT_TVI_SELL_THRESHOLD and
              tick_spread_percentage < MAX_SPREAD_PERCENTAGE):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=polled_at,
                price=latest_tick.last_price, # As per pseudocode
                rule_id=RULE_ID,
                indicators={
                    "bb_width_percentage": bb_width_percentage,
                    "tick_volume_imbalance": float(tick_volume_imbalance),
                    "tick_spread_percentage": tick_spread_percentage,
                    "hourly_sma": hourly_sma,
                    "hourly_std_dev": hourly_std_dev
                },
                confidence=1.0, # Default confidence
            ))
            
    return signals