from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from pydantic import Field

RULE_ID = "oversold_compression_reversal_v2"

# Define lookback periods for indicators
RSI_PERIOD = 14
Z_SCORE_PERIOD = 20
BBW_PERIOD = 20 # Bollinger Band Width period
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3 # %D is not used in rule logic, but kept for consistency with common Stochastic implementations
SMA_PERIOD = 5

# Minimum candles required for all indicators to produce at least one value.
# RSI(14) needs 14 + 1 = 15 candles for the first RSI value.
# Z-Score(20) needs 20 closes.
# BBW(20) needs 20 closes.
# Stochastic K(14) needs 14 candles for the first %K value.
# SMA(5) needs 5 closes.
# The maximum lookback requirement is 20 candles for Z-Score and BBW.
# Also, 'previous_close' requires at least 2 candles.
MIN_CANDLES_FOR_ALL_INDICATORS = max(
    RSI_PERIOD + 1,
    Z_SCORE_PERIOD,
    BBW_PERIOD,
    STOCH_K_PERIOD,
    SMA_PERIOD,
    2 # For previous_close
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
        # If no range, %K is 0.0 if close is at low, 100.0 if at high, 50.0 otherwise (arbitrary for flat line)
        k = 0.0 if current_close == lowest_low else (100.0 if current_close == highest_high else 50.0)
    else:
        k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
    
    return k


def calculate_sma(closes: list[float], period: int) -> float | None:
    """
    Calculates the Simple Moving Average (SMA) for a list of closing prices.
    Returns the last SMA value.
    Returns None if insufficient data.
    """
    if len(closes) < period:
        return None
    
    return statistics.mean(closes[-period:])


def calculate_bollinger_band_width(closes: list[float], period: int, num_std_dev: float = 2.0) -> float | None:
    """
    Calculates the Bollinger Band Width for a list of closing prices.
    BBW = (Upper Band - Lower Band) / Middle Band
    Returns the last BBW value.
    Returns None if insufficient data.
    """
    if len(closes) < period:
        return None
    
    recent_closes = closes[-period:]
    
    sma = statistics.mean(recent_closes)
    
    if len(recent_closes) > 1:
        std_dev = statistics.stdev(recent_closes)
    else:
        std_dev = 0.0 # Standard deviation is 0 for a single data point or identical values

    # Upper and Lower Bands
    # upper_band = sma + (std_dev * num_std_dev)
    # lower_band = sma - (std_dev * num_std_dev)
    
    # BBW = (Upper Band - Lower Band) / SMA = (2 * std_dev * num_std_dev) / SMA
    
    if sma == 0: # Avoid division by zero, especially if prices are all zero (highly unlikely)
        return 0.0 
    
    return (2 * std_dev * num_std_dev) / sma


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
        warm_closes = [c.close for c in warm_candles]
        current_close = warm_candles[-1].close
        previous_close = warm_candles[-2].close

        # --- Calculate Indicators ---
        
        # RSI(14)
        rsi_val = calculate_rsi(warm_candles, RSI_PERIOD)
        if rsi_val is None:
            continue

        # Price Z-Score (20) - using the last warm candle's close
        z_score_val = calculate_price_z_score(warm_closes, Z_SCORE_PERIOD, current_close)
        if z_score_val is None:
            continue

        # Bollinger Band Width (20)
        bb_width_val = calculate_bollinger_band_width(warm_closes, BBW_PERIOD)
        if bb_width_val is None:
            continue

        # Stochastic Oscillator (K=14) - only %K is used for this rule
        stoch_k_val = calculate_stochastic_oscillator(warm_candles, STOCH_K_PERIOD, STOCH_D_PERIOD)
        if stoch_k_val is None:
            continue
            
        # 5-period Simple Moving Average (SMA)
        sma_5_val = calculate_sma(warm_closes, SMA_PERIOD)
        if sma_5_val is None:
            continue

        # --- Entry Conditions (BuySignal) ---
        # Deeply oversold + extreme compression + initial bounce
        if (rsi_val < 30 and
            z_score_val < -2.0 and
            stoch_k_val < 10 and
            bb_width_val < 0.01 and
            current_close > previous_close):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_last_price,
                rule_id=RULE_ID,
                confidence=1.0 # Default confidence
            ))
        
        # --- Exit Conditions (SellSignal) ---
        # Overbought or short-term trend reversal
        if (rsi_val > 60 or
            current_close < sma_5_val):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_last_price,
                rule_id=RULE_ID,
                confidence=1.0 # Default confidence
            ))
            
    return signals