from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from pydantic import Field

RULE_ID = "oversold-volatility-reversal-v2"

# Define lookback periods for indicators
RSI_PERIOD = 14
ZSCORE_PERIOD = 20
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0 # Standard deviation multiplier for Bollinger Bands
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3 # %D is not used in rule logic, but required by helper function

# Minimum candles required for all indicators to produce at least one value.
# RSI(14) needs 14 + 1 = 15 candles for the first RSI value.
# Z-Score(20) needs 20 closes.
# BB Width(20) needs 20 closes.
# Stochastic K(14) needs 14 candles for the first %K value.
# The maximum lookback requirement is 20 candles for Z-Score and BB.
MIN_CANDLES_FOR_ALL_INDICATORS = max(
    RSI_PERIOD + 1,
    ZSCORE_PERIOD,
    BB_PERIOD,
    STOCH_K_PERIOD
)


def calculate_rsi(candles: list[WarmCandle], period: int) -> float | None:
    """
    Calculates the Relative Strength Index (RSI) for a list of WarmCandle objects.
    Returns the last RSI value.
    Returns None if insufficient data.
    """
    if len(candles) < period + 1:
        return None

    closes = [c.close for c in candles]
    
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]

    gains = [max(0, change) for change in changes]
    losses = [abs(min(0, change)) for change in changes]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Calculate initial RS and RSI
    if avg_loss == 0:
        rs = 1000.0 # Effectively infinity to push RSI to 100
    else:
        rs = avg_gain / avg_loss
    
    rsi_val = 100 - (100 / (1 + rs))

    # Calculate subsequent RS and RSI values using Wilder's smoothing
    for i in range(period, len(gains)):
        current_gain = gains[i]
        current_loss = losses[i]

        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period
        
        if avg_loss == 0:
            rs = 1000.0 # Effectively infinity
        else:
            rs = avg_gain / avg_loss
        
        rsi_val = 100 - (100 / (1 + rs))
    
    return rsi_val


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


def calculate_stochastic_oscillator(candles: list[WarmCandle], period_k: int, period_d: int) -> float | None:
    """
    Calculates the Stochastic Oscillator %K for the last candle.
    Returns the last %K value.
    Returns None if insufficient data.
    """
    if len(candles) < period_k:
        return None

    # We only need the latest %K value, so we calculate it for the last window
    period_candles = candles[-period_k:]
    
    highest_high = max(c.high for c in period_candles)
    lowest_low = min(c.low for c in period_candles)
    current_close = period_candles[-1].close

    if (highest_high - lowest_low) == 0:
        # If no range, %K is 0 if close is at low, 100 if at high, 50 otherwise (arbitrary for flat line)
        k = 0.0 if current_close == lowest_low else (100.0 if current_close == highest_high else 50.0)
    else:
        k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
    
    return k


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
        current_last_price = last_tick.last_price
        timestamp = last_tick.polled_at

        # --- Prepare data for indicators ---
        # The last completed hourly candle's close price
        previous_candle_close = warm_candles[-1].close 
        warm_closes = [c.close for c in warm_candles]

        # --- Calculate Indicators ---
        
        # RSI(14)
        rsi_val = calculate_rsi(warm_candles, RSI_PERIOD)
        if rsi_val is None:
            continue

        # Price Z-Score (20) - using current_last_price against warm_closes
        z_score_val = calculate_price_z_score(warm_closes, ZSCORE_PERIOD, current_last_price)
        if z_score_val is None:
            continue

        # Bollinger Band Width (20)
        bb_width_val = calculate_bollinger_band_width(warm_candles, BB_PERIOD, BB_STD_DEV_MULTIPLIER)
        if bb_width_val is None:
            continue

        # Stochastic Oscillator (K=14) - only %K is used for this rule
        stoch_k_val = calculate_stochastic_oscillator(warm_candles, STOCH_K_PERIOD, STOCH_D_PERIOD)
        if stoch_k_val is None:
            continue

        # --- Entry Conditions (BuySignal) ---
        # 1. Broadened oversold conditions
        oversold_rsi = rsi_val < 30
        oversold_stoch = stoch_k_val < 20
        undervalued_zscore = z_score_val < -1.5

        # 2. Very low volatility (price compression)
        low_volatility = bb_width_val < 0.01

        # 3. Bounce confirmation (current price shows a positive move from previous candle)
        bounce_confirmation = current_last_price > previous_candle_close

        if oversold_rsi and oversold_stoch and undervalued_zscore and low_volatility and bounce_confirmation:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_last_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))
        
        # --- Exit Conditions (SellSignal) ---
        # 1. RSI recovery from oversold levels, indicating momentum shift
        if rsi_val > 40:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_last_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))
            
    return signals