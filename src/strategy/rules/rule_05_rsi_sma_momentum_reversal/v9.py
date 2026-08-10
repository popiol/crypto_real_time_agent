from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from pydantic import Field

RULE_ID = "tiered_oversold_entry_v1"

# Define lookback periods for indicators
RSI_PERIOD = 14
ZSCORE_PERIOD = 20
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0 # Standard 2 standard deviations for Bollinger Bands

# Sell thresholds (retained from original rule, as only entry logic was modified)
RSI_EXIT_SELL = 60
ZSCORE_EXIT_SELL = 0.5

# Minimum candles required for all indicators to produce at least one value.
# RSI(14) needs 14 + 1 = 15 candles for the first RSI value.
# Z-Score(20) needs 20 candles.
# BB Width(20) needs 20 candles.
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
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]

    # Separate gains and losses
    gains = [max(0, change) for change in changes]
    losses = [abs(min(0, change)) for change in changes]

    avg_gains = []
    avg_losses = []
    rsi_values = []

    # Calculate initial average gain/loss over the first 'period' changes
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
        # If no volatility, Z-score is 0.0 if current_price is at the mean, otherwise approaches infinity.
        # For trading, 0.0 implies current_price is at the mean.
        return 0.0 if current_price == mean_closes else (1.0 if current_price > mean_closes else -1.0)
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

        # --- Prepare data for indicators ---
        warm_closes = [c.close for c in warm_candles]

        # --- Calculate Indicators ---
        
        # RSI(14)
        rsi_values = calculate_rsi(warm_candles, RSI_PERIOD)
        if rsi_values is None or not rsi_values: # Need at least one RSI value
            continue
        current_rsi = rsi_values[-1]

        # Price Z-Score (20) - using the latest tick price for responsiveness
        price_z_score_current = calculate_price_z_score(warm_closes, ZSCORE_PERIOD, last_price)
        if price_z_score_current is None:
            continue

        # Bollinger Band Width (20)
        current_bb_width = calculate_bollinger_band_width(warm_candles, BB_PERIOD, BB_STD_DEV_MULTIPLIER)
        if current_bb_width is None:
            continue
        
        # --- BUY Signal Conditions (Tiered Approach) ---

        # Condition 1: High conviction, low volatility reversal
        # RSI < 15, Z-Score < -1.5, BB Width < 0.01
        buy_condition_1 = (
            (current_rsi < 15) and
            (price_z_score_current < -1.5) and
            (current_bb_width < 0.01)
        )

        # Condition 2: High conviction, moderate volatility reversal
        # RSI < 25, Z-Score < -2.0, BB Width between 0.05 and 0.5
        buy_condition_2 = (
            (current_rsi < 25) and
            (price_z_score_current < -2.0) and
            (current_bb_width >= 0.05) and
            (current_bb_width <= 0.5)
        )

        if buy_condition_1 or buy_condition_2:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=last_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence as strict conditions are met
            ))
        
        # --- SELL Signal Conditions (to close an existing long position) ---
        # 1. RSI(14) rises above 60 (significant recovery)
        # 2. Price Z-Score(20) exceeds 0.5 (price has reverted significantly above the mean)
        sell_condition = (
            (current_rsi > RSI_EXIT_SELL) or
            (price_z_score_current > ZSCORE_EXIT_SELL)
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