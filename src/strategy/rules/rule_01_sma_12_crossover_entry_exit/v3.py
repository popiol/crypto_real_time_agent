from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal

# Rule ID
RULE_ID = "sma12_momentum_filter_v2"

# Constants for indicator periods and thresholds
SMA_PERIOD = 12
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STDDEV = 2

# Buy signal thresholds
MIN_RSI_BUY = 50
MIN_BB_PERCENT_B_BUY = 0.5
MAX_SPREAD_PCT = 0.01  # 1% spread

# Sell signal (exit long) thresholds
RSI_EXIT_THRESHOLD = MIN_RSI_BUY - 5  # 45
BB_PERCENT_B_EXIT_THRESHOLD = MIN_BB_PERCENT_B_BUY - 0.1  # 0.4

# Minimum data requirements for indicators
# SMA_PERIOD = 12
# RSI_PERIOD = 14 (needs 14+1=15 candles for first RSI value)
# BB_PERIOD = 20
MIN_WARM_CANDLES_REQUIRED = max(SMA_PERIOD, RSI_PERIOD + 1, BB_PERIOD)
MIN_HOT_TICKS = 1  # Only the most recent tick is needed for current_price and spread

# Helper functions for indicator calculations
def calculate_sma(closes: list[float], period: int) -> float | None:
    """Calculates the Simple Moving Average."""
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))

def calculate_rsi(closes: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI)."""
    if len(closes) < period + 1:
        return None

    price_array = np.array(closes, dtype=float)
    deltas = np.diff(price_array)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(gains)
    avg_loss = np.zeros_like(losses)

    # Initial average gain/loss over the first 'period' deltas
    if len(gains) < period:
        return None

    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    # Smoothed average gain/loss for subsequent periods
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

    last_avg_gain = avg_gain[-1]
    last_avg_loss = avg_loss[-1]

    if last_avg_loss == 0:
        return 100.0 if last_avg_gain > 0 else 0.0
    
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
    Implements the Refined SMA-12 Crossover with Momentum and Liquidity Filters rule.

    A buy signal requires the current price to be above the 12-period SMA, 
    RSI > 50, %B > 0.5, and a low bid-ask spread.
    A sell signal (to exit a long) is triggered when the price falls below the 12-period SMA
    or momentum weakens significantly (RSI < 45 or %B < 0.4).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm data for all indicators
        if len(pair_data.warm) < MIN_WARM_CANDLES_REQUIRED:
            continue

        # Ensure enough hot data for current price and spread
        if not pair_data.hot:
            continue

        # Extract relevant data
        hourly_closes = [candle.close for candle in pair_data.warm]
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_tick_timestamp = current_tick.polled_at

        # --- Calculate Indicators ---
        # SMA(12)
        sma_12 = calculate_sma(hourly_closes, SMA_PERIOD)
        if sma_12 is None: continue # Should not happen if MIN_WARM_CANDLES_REQUIRED is met

        price_above_sma_12 = current_price > sma_12

        # RSI(14) - based on hourly closes
        current_rsi = calculate_rsi(hourly_closes, RSI_PERIOD)
        if current_rsi is None: continue # Should not happen if MIN_WARM_CANDLES_REQUIRED is met

        # Bollinger Bands (20, 2) and %B - BB are based on hourly closes, %B uses current_price
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(hourly_closes, BB_PERIOD, BB_STDDEV)
        
        # Handle cases where BB cannot be calculated or bands are collapsed (std_dev = 0)
        if bb_upper is None or bb_lower is None or (bb_upper - bb_lower) == 0: 
            continue 

        percent_b = (current_price - bb_lower) / (bb_upper - bb_lower)

        # Bid-Ask Spread Percentage - using current tick data
        # Ensure bid_price is not zero to prevent division by zero
        if current_tick.bid_price <= 0:
            continue
        bid_ask_spread_pct = (current_tick.ask_price - current_tick.bid_price) / current_tick.bid_price
        
        # Store indicators for potential signal
        indicators = {
            "sma_12": sma_12,
            "current_rsi": current_rsi,
            "percent_b": percent_b,
            "bid_ask_spread_pct": bid_ask_spread_pct,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
        }

        # --- Entry Condition (Long Only) ---
        # The agent handles whether a position is already open. This rule just emits a BuySignal.
        if (price_above_sma_12 and
            current_rsi > MIN_RSI_BUY and
            percent_b > MIN_BB_PERCENT_B_BUY and
            bid_ask_spread_pct < MAX_SPREAD_PCT):
            
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=current_tick_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    indicators=indicators
                )
            )
        # --- Exit Condition (Close Long Position) ---
        # The agent handles whether a position is open. This rule just emits a SellSignal.
        # This is for closing an existing long position.
        elif (not price_above_sma_12 or # Price falls below SMA
              current_rsi < RSI_EXIT_THRESHOLD or # Momentum weakens
              percent_b < BB_PERCENT_B_EXIT_THRESHOLD): # Price drops within BB range
            
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=current_tick_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    exit_reason="SMA_Momentum_Breakdown",
                    indicators=indicators
                )
            )

    return signals