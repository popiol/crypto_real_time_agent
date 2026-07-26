from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, Tick, WarmCandle
from datetime import datetime

# Rule ID
RULE_ID = "oversold_mean_reversion_exit_refinement"

# Constants for indicator periods and thresholds
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STDDEV = 2.0  # Standard deviation multiplier for Bollinger Bands
SMA_PERIOD = 12

# Entry thresholds
RSI_ENTRY_THRESHOLD_1 = 20.0 # First condition
PRICE_TO_SMA_DEV_ENTRY_THRESHOLD_1 = -1.5 # First condition
BB_PERCENT_B_ENTRY_THRESHOLD_1 = 0.1 # First condition
BID_ASK_SPREAD_PCT_MAX_1 = 2.0 # First condition spread filter

RSI_ENTRY_THRESHOLD_2 = 5.0 # Second (extreme) condition
PRICE_TO_SMA_DEV_ENTRY_THRESHOLD_2 = -3.0 # Second (extreme) condition
BID_ASK_SPREAD_PCT_MIN_2 = 2.0 # Second (extreme) condition spread filter

# Exit parameters (stored in BuySignal for agent, and fixed for rule-generated SellSignal)
PROFIT_TARGET_PCT = 1.5
STOP_LOSS_PCT = 2.5
RSI_EXIT_THRESHOLD = 40.0 # Used for rule-generated SellSignal
MAX_HOLD_HOURS = 12.0

# Minimum data requirements for indicators
# RSI_PERIOD = 14 requires 14+1=15 candles for the first RSI value.
# BB_PERIOD = 20 requires 20 candles.
# SMA_PERIOD = 12 requires 12 candles.
# The pseudocode specifies 20, which covers all.
MIN_WARM_CANDLES_REQUIRED = max(RSI_PERIOD + 1, BB_PERIOD, SMA_PERIOD)
MIN_HOT_TICKS = 1  # Only the most recent tick is needed for current_price and spread

# Helper functions for indicator calculations
def calculate_sma(closes: list[float], period: int) -> float | None:
    """Calculates the Simple Moving Average."""
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))

def calculate_rsi(closes: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI)."""
    if len(closes) < period + 1: # Need period + 1 closes for 'period' deltas
        return None

    price_array = np.array(closes, dtype=float)
    deltas = np.diff(price_array)
    
    # Ensure we have enough deltas for the initial average calculation
    if len(deltas) < period:
        return None

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(gains)
    avg_loss = np.zeros_like(losses)

    # Initial average gain/loss over the first 'period' deltas
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    # Smoothed average gain/loss for subsequent periods
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

    last_avg_gain = avg_gain[-1]
    last_avg_loss = avg_loss[-1]

    if last_avg_loss == 0:
        return 100.0 if last_avg_gain > 0 else 0.0 # Handle division by zero for RS
    
    rs = last_avg_gain / last_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(closes: list[float], period: int, stddev_multiplier: float) -> tuple[float | None, float | None, float | None]:
    """Calculates Bollinger Bands (Upper, Middle, Lower)."""
    if len(closes) < period:
        return None, None, None

    closes_arr = np.array(closes[-period:], dtype=float)
    
    middle_band = float(np.mean(closes_arr))
    std_dev = float(np.std(closes_arr))

    upper_band = middle_band + (std_dev * stddev_multiplier)
    lower_band = middle_band - (std_dev * stddev_multiplier)

    return upper_band, middle_band, lower_band

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Refined Oversold Mean Reversion with Dynamic Exits rule.
    
    Emits a BuySignal for deeply oversold assets based on RSI, price deviation from SMA,
    and Bollinger Band %B, with a liquidity filter. The BuySignal includes parameters
    for profit target, stop loss, RSI recovery, and time-based exits, which are intended
    to be used by the trading agent for position management.
    
    Emits a SellSignal (to close an existing long position) when RSI recovers above a
    predefined threshold (RSI_EXIT_THRESHOLD). Profit target, stop loss, and time-based
    exits are intended to be handled by the trading agent that consumes these signals
    and manages open positions, as they require position-specific context not available
    in the `MarketData` object.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm data for all indicators
        if len(pair_data.warm) < MIN_WARM_CANDLES_REQUIRED:
            continue

        # Ensure enough hot data for current price and spread calculation
        if not pair_data.hot or len(pair_data.hot) < MIN_HOT_TICKS:
            continue

        # Extract relevant data
        hourly_closes = [candle.close for candle in pair_data.warm]
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_tick_timestamp = current_tick.polled_at

        # --- Calculate Indicators ---
        sma_12 = calculate_sma(hourly_closes, SMA_PERIOD)
        if sma_12 is None or sma_12 == 0: # Ensure SMA is calculable and not zero for deviation calc
            continue

        current_rsi = calculate_rsi(hourly_closes, RSI_PERIOD)
        if current_rsi is None:
            continue

        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(hourly_closes, BB_PERIOD, BB_STDDEV)
        
        # Handle cases where BB cannot be calculated or bands are collapsed (std_dev = 0)
        if bb_upper is None or bb_lower is None: 
            continue
        
        bb_range = bb_upper - bb_lower
        if bb_range == 0: # If std_dev is 0, bands are collapsed, %B is not meaningful
            percent_b = 0.0 # Setting to 0.0 effectively makes it below 0.1 threshold
        else:
            percent_b = (current_price - bb_lower) / bb_range

        price_to_sma_dev = (current_price - sma_12) / sma_12 * 100

        # Bid-ask spread percentage
        if current_tick.bid_price > 0: # Ensure bid_price is not zero to avoid division by zero
            bid_ask_spread_pct = (current_tick.ask_price - current_tick.bid_price) / current_tick.bid_price * 100
        else:
            bid_ask_spread_pct = float('inf') # Effectively fail the spread filter

        # Store indicators for potential signal
        # These are the *current* indicator values at the time of signal generation.
        indicators = {
            "sma_12": sma_12,
            "current_rsi": current_rsi,
            "percent_b": percent_b,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "price_to_sma_dev": price_to_sma_dev,
            "bid_ask_spread_pct": bid_ask_spread_pct,
        }

        # --- Entry Conditions (Long Only) ---
        # 1. Primary Entry Conditions
        is_primary_entry_condition = (
            current_rsi < RSI_ENTRY_THRESHOLD_1 and
            price_to_sma_dev < PRICE_TO_SMA_DEV_ENTRY_THRESHOLD_1 and
            percent_b < BB_PERCENT_B_ENTRY_THRESHOLD_1
        )
        
        # 2. Extreme Conditions Entry (allows higher spread)
        is_extreme_entry_condition = (
            bid_ask_spread_pct >= BID_ASK_SPREAD_PCT_MIN_2 and # Higher spread allowed
            current_rsi < RSI_ENTRY_THRESHOLD_2 and # More extreme RSI
            price_to_sma_dev < PRICE_TO_SMA_DEV_ENTRY_THRESHOLD_2 # More extreme deviation
        )

        if is_primary_entry_condition and bid_ask_spread_pct < BID_ASK_SPREAD_PCT_MAX_1:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=current_tick_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    indicators={
                        **indicators, # Include calculated indicators
                        "profit_target_pct": PROFIT_TARGET_PCT,
                        "stop_loss_pct": STOP_LOSS_PCT,
                        "rsi_exit_threshold_buy": RSI_EXIT_THRESHOLD, # Renamed to avoid conflict if agent uses "rsi_exit_threshold" for internal logic
                        "max_hold_hours": MAX_HOLD_HOURS,
                        "entry_type": "primary"
                    }
                )
            )
        elif is_extreme_entry_condition:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=current_tick_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    indicators={
                        **indicators, # Include calculated indicators
                        "profit_target_pct": PROFIT_TARGET_PCT,
                        "stop_loss_pct": STOP_LOSS_PCT,
                        "rsi_exit_threshold_buy": RSI_EXIT_THRESHOLD,
                        "max_hold_hours": MAX_HOLD_HOURS,
                        "entry_type": "extreme_conditions"
                    }
                )
            )
        
        # --- Exit Condition (Close Long Position) ---
        # This SellSignal is emitted if the RSI recovers above a fixed threshold.
        # It is intended to close an existing long position opened by THIS rule.
        # The agent consuming this signal will decide whether a position exists.
        elif current_rsi > RSI_EXIT_THRESHOLD:
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=current_tick_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    exit_reason="RSI_Recovery_Exit",
                    indicators=indicators # Include current indicator values at exit
                )
            )

    return signals