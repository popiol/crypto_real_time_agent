from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from pydantic import Field # Needed for Field in models, though not directly used in the rule logic itself

RULE_ID = "RSI_ZScore_Dip_V2_Relaxed"

# Define lookback periods for indicators
RSI_PERIOD = 14
ZSCORE_PERIOD = 20
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0 # Standard 2 standard deviations for Bollinger Bands

# Minimum candles required for all indicators to produce at least one value.
# RSI(14) needs 14 + 1 = 15 candles for the first RSI value.
# Z-Score(20) needs 20 candles for mean/std dev.
# BB Width(20) needs 20 candles for mean/std dev.
MIN_CANDLES_FOR_ALL_INDICATORS = max(RSI_PERIOD + 1, ZSCORE_PERIOD, BB_PERIOD)

def calculate_rsi(candles: list[WarmCandle], period: int) -> list[float] | None:
    """
    Calculates the Relative Strength Index (RSI) for a list of WarmCandle objects.
    Returns a list of RSI values.
    Returns None if insufficient data.
    """
    # Need at least period + 1 candles for the first RSI value.
    if len(candles) < period + 1:
        return None

    closes = [c.close for c in candles]
    
    # Calculate price changes
    # changes[i] is the change from closes[i] to closes[i+1]
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]

    # Separate gains and losses
    gains = [max(0, change) for change in changes]
    losses = [abs(min(0, change)) for change in changes]

    avg_gains = []
    avg_losses = []
    rsi_values = []

    # Calculate initial average gain/loss over the first 'period' changes
    # The first 'period' changes correspond to candles[1]...candles[period]
    # So we use gains[:period] and losses[:period]
    initial_avg_gain = sum(gains[:period]) / period
    initial_avg_loss = sum(losses[:period]) / period
    avg_gains.append(initial_avg_gain)
    avg_losses.append(initial_avg_loss)

    # Calculate the first RSI value (for the candle at index 'period')
    if initial_avg_loss == 0:
        rs = 1000.0 # Effectively infinity to push RSI to 100
    else:
        rs = initial_avg_gain / initial_avg_loss
    rsi_values.append(100 - (100 / (1 + rs)))

    # Calculate subsequent smoothed averages and RSI values
    # We start from the (period)-th change to calculate RSI for candle (period + 1)
    for i in range(period, len(gains)):
        current_gain = gains[i]
        current_loss = losses[i]

        next_avg_gain = (avg_gains[-1] * (period - 1) + current_gain) / period
        next_avg_loss = (avg_losses[-1] * (period - 1) + current_loss) / period
        
        avg_gains.append(next_avg_gain)
        avg_losses.append(next_avg_loss)

        if next_avg_loss == 0:
            rs = 1000.0 # Effectively infinity
        else:
            rs = next_avg_gain / next_avg_loss
        
        rsi_values.append(100 - (100 / (1 + rs)))
    
    return rsi_values

def calculate_price_z_score(closes: list[float], period: int, current_price: float) -> float | None:
    """
    Calculates the Z-Score of the current_price relative to the mean and standard deviation
    of the recent 'period' closing prices.
    Returns None if insufficient data.
    """
    if len(closes) < period:
        return None
    
    # Take the last 'period' closes for calculation
    recent_closes = closes[-period:]

    if not recent_closes: 
        return None

    mean_closes = statistics.mean(recent_closes)
    
    # Handle case where all prices are the same (no volatility)
    if len(recent_closes) > 1:
        std_dev_closes = statistics.stdev(recent_closes)
    else: # If only one data point, standard deviation is 0
        std_dev_closes = 0.0

    if std_dev_closes == 0:
        # If no volatility, Z-score is 0.0 if current_price is at the mean, otherwise undefined/infinity.
        # For trading, 0.0 implies current_price is at the mean.
        return 0.0 if current_price == mean_closes else (1.0 if current_price > mean_closes else -1.0) # Or handle as None/NaN
    else:
        return (current_price - mean_closes) / std_dev_closes

def calculate_bollinger_band_width(candles: list[WarmCandle], period: int, std_dev_multiplier: float) -> float | None:
    """
    Calculates the Bollinger Band Width as (Upper Band - Lower Band) / Middle Band (SMA).
    Returns None if insufficient data.
    """
    if len(candles) < period:
        return None

    closes = [c.close for c in candles[-period:]] # Use only the last 'period' closes

    if not closes:
        return None

    sma = statistics.mean(closes)
    
    # Calculate standard deviation for the last 'period' closes
    if len(closes) > 1:
        std_dev = statistics.stdev(closes)
    else:
        std_dev = 0.0 # If only one data point, standard deviation is 0

    # If SMA is zero (e.g., for non-price data, or highly unusual scenario), or no volatility
    if sma == 0 or std_dev == 0:
        return 0.0 # No width if no price or no volatility

    upper_band = sma + (std_dev * std_dev_multiplier)
    lower_band = sma - (std_dev * std_dev_multiplier)
    
    bb_width = (upper_band - lower_band) / sma # Relative width
    return bb_width


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Ensure enough warm candles for all indicators
        if len(warm_candles) < MIN_CANDLES_FOR_ALL_INDICATORS:
            continue

        # Ensure hot data exists for current price and timestamp
        if not hot_ticks:
            continue
        
        last_tick = hot_ticks[-1]
        last_price = last_tick.last_price
        timestamp = last_tick.polled_at
        current_bid = last_tick.bid_price
        current_ask = last_tick.ask_price

        # --- Prepare data for indicators ---
        warm_closes = [c.close for c in warm_candles]

        # --- Calculate Indicators ---
        
        # RSI(14)
        rsi_values = calculate_rsi(warm_candles, RSI_PERIOD)
        if rsi_values is None or not rsi_values: # Ensure at least one RSI value
            continue
        current_rsi = rsi_values[-1]

        # Price Z-Score (20) - using last_price from hot ticks as current price
        price_z_score_current = calculate_price_z_score(warm_closes, ZSCORE_PERIOD, last_price)
        if price_z_score_current is None:
            continue

        # Bollinger Band Width (20)
        current_bb_width = calculate_bollinger_band_width(warm_candles, BB_PERIOD, BB_STD_DEV_MULTIPLIER)
        if current_bb_width is None:
            continue

        # Bid-Ask Spread Percentage
        current_spread_pct = 0.0
        # Use last_tick.mid_price for spread_rel, which is already calculated in Tick,
        # or calculate explicitly using last_price as denominator as per pseudocode.
        # Pseudocode: (latest_tick.ask - latest_tick.bid) / latest_tick.last_price * 100
        if last_price > 0: # Avoid division by zero
            current_spread_pct = ((current_ask - current_bid) / last_price) * 100
        else:
            continue # Cannot calculate spread percentage if last_price is zero
        
        # --- BUY Signal Conditions (Relaxed and Enhanced) ---
        buy_condition = (
            20 < current_rsi < 40 and                   # Relaxed RSI range
            price_z_score_current < -0.5 and            # Relaxed Z-Score threshold
            current_spread_pct < 0.75 and               # Strict liquidity filter
            0.01 < current_bb_width < 0.15              # Moderate volatility filter
        )

        if buy_condition:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=last_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence as conditions are met
            ))
        
        # --- SELL Signal Conditions (Market-derived Exit) ---
        sell_condition = (
            current_rsi > 60 or                         # Overbought
            price_z_score_current > 1.0 or              # Significantly overvalued
            current_spread_pct > 1.5                    # Illiquid market exit
        )

        if sell_condition:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=last_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence as conditions are met
            ))
            
    return signals