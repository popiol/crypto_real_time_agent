from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, Tick, WarmCandle

# Parameters
TICK_VOLUME_IMBALANCE_THRESHOLD = 1000
HOURLY_BB_WIDTH_MAX_PERCENT = 0.10  # Max 0.10% for mean reversion
TICK_SPREAD_MIN_PERCENT = 0.5
TICK_SPREAD_MAX_PERCENT = 7.8
BB_LOOKBACK_PERIOD = 20  # Fits data.warm (max 24 candles)
STOP_LOSS_STD_DEV_MULTIPLIER = 1.5
RULE_ID = "mean_reversion_filtered_v2"

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # --- Data Acquisition & Validation ---
        if not pair_data.hot:
            continue
        
        latest_tick: Tick = pair_data.hot[-1]
        polled_at = latest_tick.polled_at

        # Tick Volume Imbalance:
        # Interpreting 'Tick Volume Imbalance' as the difference between best bid volume
        # and best ask volume from the latest tick, as these are the only volume
        # related fields available on the Tick model for an 'imbalance'.
        tick_volume_imbalance = latest_tick.bid_volume - latest_tick.ask_volume
        
        # Tick Bid-Ask Spread Percentage: Directly available
        tick_bid_ask_spread_percentage = latest_tick.spread_rel

        # Hourly Indicator Calculation (from data.warm)
        if len(pair_data.warm) < BB_LOOKBACK_PERIOD:
            continue
        
        # Get the last BB_LOOKBACK_PERIOD closing prices
        hourly_closes = np.array([c.close for c in pair_data.warm[-BB_LOOKBACK_PERIOD:]])

        # Calculate SMA and Standard Deviation
        hourly_sma = np.mean(hourly_closes)
        # Using population standard deviation (ddof=0) for Bollinger Bands
        hourly_std_dev = np.std(hourly_closes, ddof=0)

        # Handle cases where SMA might be zero (e.g., if all prices are zero, which is unlikely for real data)
        if hourly_sma <= 0:
            continue # Cannot calculate percentage width or std_dev_as_percent

        # Calculate Bollinger Bands Width
        # Assuming 2-sigma bands as per pseudocode comment "Assuming 2-sigma bands"
        hourly_upper_band = hourly_sma + (hourly_std_dev * 2)
        hourly_lower_band = hourly_sma - (hourly_std_dev * 2)
            
        hourly_bb_width_percent = ((hourly_upper_band - hourly_lower_band) / hourly_sma) * 100

        # --- Rule Logic ---
        is_bb_width_ok = hourly_bb_width_percent < HOURLY_BB_WIDTH_MAX_PERCENT
        is_spread_ok = (tick_bid_ask_spread_percentage >= TICK_SPREAD_MIN_PERCENT and
                        tick_bid_ask_spread_percentage <= TICK_SPREAD_MAX_PERCENT)

        # Calculate std_dev as a percentage of the price level (relative volatility)
        std_dev_as_percent = hourly_std_dev / hourly_sma
        
        # BUY Signal
        if (tick_volume_imbalance < -TICK_VOLUME_IMBALANCE_THRESHOLD and
                is_bb_width_ok and
                is_spread_ok):
            
            entry_price = latest_tick.ask_price
            
            stop_loss_price = entry_price * (1 - (STOP_LOSS_STD_DEV_MULTIPLIER * std_dev_as_percent))
            take_profit_price = hourly_sma # Target the middle band for mean reversion
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=polled_at,
                price=entry_price,
                rule_id=RULE_ID,
                indicators={
                    "tick_volume_imbalance": tick_volume_imbalance,
                    "hourly_bb_width_percent": hourly_bb_width_percent,
                    "tick_bid_ask_spread_percentage": tick_bid_ask_spread_percentage,
                    "hourly_sma": hourly_sma,
                    "hourly_std_dev": hourly_std_dev,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price
                }
            ))

        # SELL Signal
        elif (tick_volume_imbalance > TICK_VOLUME_IMBALANCE_THRESHOLD and
              is_bb_width_ok and
              is_spread_ok):
            
            entry_price = latest_tick.bid_price
            
            stop_loss_price = entry_price * (1 + (STOP_LOSS_STD_DEV_MULTIPLIER * std_dev_as_percent))
            take_profit_price = hourly_sma # Target the middle band for mean reversion
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=polled_at,
                price=entry_price,
                rule_id=RULE_ID,
                indicators={
                    "tick_volume_imbalance": tick_volume_imbalance,
                    "hourly_bb_width_percent": hourly_bb_width_percent,
                    "tick_bid_ask_spread_percentage": tick_bid_ask_spread_percentage,
                    "hourly_sma": hourly_sma,
                    "hourly_std_dev": hourly_std_dev,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price
                }
            ))
            
    return signals