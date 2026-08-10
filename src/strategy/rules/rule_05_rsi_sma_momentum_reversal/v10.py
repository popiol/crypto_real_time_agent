from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from pydantic import Field

RULE_ID = "relaxed_extreme_oversold_reversal_v1"

# Define lookback periods for indicators
RSI_PERIOD = 14
ZSCORE_PERIOD = 20
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0
STOCH_K_PERIOD = 14
# STOCH_D_PERIOD is not explicitly used in the rule's logic, but the helper function
# requires it for calculating D values (which are then ignored by this rule).
# We'll use a common default value.
STOCH_D_PERIOD = 3 

# Minimum candles required for all indicators to produce at least one value.
# RSI(14) needs 14 + 1 = 15 candles for the first RSI value.
# Z-Score(20) needs 20 candles.
# BB Width(20) needs 20 candles.
# Stochastic K(14) needs 14 candles for the first %K value.
# The maximum lookback requirement is 20 candles.
MIN_CANDLES_FOR_ALL_INDICATORS = max(
    RSI_PERIOD + 1,
    ZSCORE_PERIOD,
    BB_PERIOD,
    STOCH_K_PERIOD
)


def calculate_rsi(candles: list[WarmCandle], period: int) -> list[float] | None:
    """
    Calculates the Relative Strength Index (RSI) for a list of WarmCandle objects.
    Returns a list of RSI values.
    Returns None if insufficient data.
    """
    if len(candles) < period + 1:
        return None

    closes = [c.close for c in candles]
    
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]

    gains = [max(0, change) for change in changes]
    losses = [abs(min(0, change)) for change in changes]

    avg_gains = []
    avg_losses = []
    rsi_values = []

    initial_avg_gain = sum(gains[:period]) / period
    initial_avg_loss = sum(losses[:period]) / period
    avg_gains.append(initial_avg_gain)
    avg_losses.append(initial_avg_loss)

    if initial_avg_loss == 0:
        rs = 1000.0 # Effectively infinity to push RSI to 100
    else:
        rs = initial_avg_gain / initial_avg_loss
    rsi_values.append(100 - (100 / (1 + rs)))

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
    
    recent_closes = closes[-period:]

    if not recent_closes: 
        return None

    mean_closes = statistics.mean(recent_closes)
    
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


def calculate_stochastic_oscillator(candles: list[WarmCandle], period_k: int, period_d: int) -> tuple[list[float] | None, list[float] | None]:
    """
    Calculates the Stochastic Oscillator (%K and %D).
    Returns a tuple of lists (k_values, d_values).
    Returns (None, None) if insufficient data.
    """
    if len(candles) < period_k:
        return None, None

    k_values = []
    for i in range(period_k - 1, len(candles)):
        period_candles = candles[i - period_k + 1 : i + 1]
        
        highest_high = max(c.high for c in period_candles)
        lowest_low = min(c.low for c in period_candles)
        current_close = period_candles[-1].close

        if (highest_high - lowest_low) == 0:
            # If no range, %K is 0 if close is at low, 100 if at high, 50 otherwise (arbitrary for flat line)
            k = 0.0 if current_close == lowest_low else (100.0 if current_close == highest_high else 50.0)
        else:
            k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        k_values.append(k)
    
    if len(k_values) < period_d:
        return k_values, None # Not enough K values to compute D

    # Calculate %D as a Simple Moving Average of %K
    d_values = []
    for i in range(period_d - 1, len(k_values)):
        d = statistics.mean(k_values[i - period_d + 1 : i + 1])
        d_values.append(d)
        
    return k_values, d_values


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
        current_price = last_tick.last_price
        timestamp = last_tick.polled_at

        # --- Prepare data for indicators ---
        warm_closes = [c.close for c in warm_candles]
        last_candle_close = warm_closes[-1]

        # --- Calculate Indicators ---
        
        # RSI(14)
        rsi_values = calculate_rsi(warm_candles, RSI_PERIOD)
        if rsi_values is None or not rsi_values:
            continue
        current_rsi = rsi_values[-1]

        # Price Z-Score (20) - using the last completed candle's close price as per pseudocode
        price_z_score = calculate_price_z_score(warm_closes, ZSCORE_PERIOD, last_candle_close)
        if price_z_score is None:
            continue

        # Bollinger Band Width (20)
        bb_width = calculate_bollinger_band_width(warm_candles, BB_PERIOD, BB_STD_DEV_MULTIPLIER)
        if bb_width is None:
            continue

        # Stochastic Oscillator (K=14, D=3) - only %K is used for this rule
        stoch_k_values, _ = calculate_stochastic_oscillator(warm_candles, STOCH_K_PERIOD, STOCH_D_PERIOD)
        if stoch_k_values is None or not stoch_k_values:
            continue
        current_stoch_k = stoch_k_values[-1]

        # --- BUY Signal Conditions (Relaxed Extreme Oversold Reversal with BB Contraction) ---
        
        # Entry Condition:
        # Price Z-Score < -2.0
        # AND (RSI < 30 OR Stochastic K < 15)
        # AND BB Width < 0.01
        buy_condition = (
            price_z_score < -2.0 and
            (current_rsi < 30 or current_stoch_k < 15) and
            bb_width < 0.01
        )

        if buy_condition:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))
        
        # --- SELL Signal Conditions (Exit existing long position) ---
        
        # Exit Condition:
        # RSI > 50
        # OR Price Z-Score > -0.5
        sell_condition = (
            current_rsi > 50 or 
            price_z_score > -0.5
        )

        if sell_condition:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))
            
    return signals