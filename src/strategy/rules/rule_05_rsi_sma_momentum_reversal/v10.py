from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

RULE_ID = "oversold_bounce_fix_v1"

# Define lookback periods for indicators
RSI_PERIOD = 14
Z_SCORE_PERIOD = 20
STOCH_K_PERIOD = 14
SMA_PERIOD = 5

# Minimum candles required for all indicators to produce at least one value.
# RSI(14) needs 14+1 = 15 candles.
# Z-Score(20) needs 20 closes.
# Stochastic K(14) needs 14 candles.
# SMA(5) needs 5 closes.
MIN_CANDLES_REQUIRED = max(RSI_PERIOD + 1, Z_SCORE_PERIOD, STOCH_K_PERIOD, SMA_PERIOD)


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

    # Initial average gain/loss
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Calculate initial RS and RSI
    if avg_loss == 0:
        rs = 1000.0 # Effectively infinity to push RSI to 100
    else:
        rs = avg_gain / avg_loss
    
    rsi_val = 100 - (100 / (1 + rs))

    # Calculate subsequent RS and RSI values using Wilder's smoothing
    # The loop should start from 'period' index in 'gains' and 'losses' lists
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
        return None # Should be caught by len(closes) < period check

    mean_closes = statistics.mean(recent_closes)
    
    if len(recent_closes) > 1:
        std_dev_closes = statistics.stdev(recent_closes)
    else: # If only one data point or all identical, standard deviation is 0
        std_dev_closes = 0.0

    if std_dev_closes == 0:
        # If no volatility, Z-score is 0.0 if current_price is at the mean, otherwise approaches infinity.
        # For trading, 0.0 implies current_price is at the mean.
        return 0.0 if current_price == mean_closes else (1.0 if current_price > mean_closes else -1.0)
    else:
        return (current_price - mean_closes) / std_dev_closes


def calculate_stochastic_oscillator(candles: list[WarmCandle], period_k: int, period_d: int = 3) -> float | None:
    """
    Calculates the Stochastic Oscillator %K for the last candle.
    The period_d parameter is included for consistency but %D is not calculated or returned.
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
        # If no range (highest_high == lowest_low), %K is 0.0 if close is at low, 100.0 if at high, 50.0 otherwise.
        # This handles flat periods gracefully, preventing division by zero.
        k = 0.0 if current_close == lowest_low else (100.0 if current_close == highest_high else 50.0)
    else:
        k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
    
    return k

def calculate_sma(candles: list[WarmCandle], period: int) -> float | None:
    """
    Calculates the Simple Moving Average (SMA) for the last 'period' closing prices.
    Returns None if insufficient data.
    """
    if len(candles) < period:
        return None
    
    closes = [c.close for c in candles[-period:]]
    return statistics.mean(closes)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Ensure enough warm candles for all indicators
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Ensure hot data exists for current price and timestamp
        if not hot_ticks:
            continue
        
        last_tick = hot_ticks[-1]
        current_last_price = last_tick.last_price
        timestamp = last_tick.polled_at

        # --- Prepare data for indicators ---
        warm_closes = [c.close for c in warm_candles]
        
        # --- Calculate Indicators ---
        
        # Current RSI(14)
        current_rsi = calculate_rsi(warm_candles, RSI_PERIOD)
        if current_rsi is None: # Should not happen if MIN_CANDLES_REQUIRED check passes
            continue
        
        # Price Z-Score (20) - using the last warm candle's close for the calculation base
        current_z_score = calculate_price_z_score(warm_closes, Z_SCORE_PERIOD, warm_candles[-1].close)
        if current_z_score is None:
            continue

        # Current Stochastic Oscillator %K (14)
        current_stoch_k = calculate_stochastic_oscillator(warm_candles, STOCH_K_PERIOD)
        if current_stoch_k is None:
            continue
        
        # SMA(5)
        sma_5 = calculate_sma(warm_candles, SMA_PERIOD)
        if sma_5 is None:
            continue

        # --- Entry Conditions (BuySignal) ---
        # Extreme oversold conditions (RSI < 20, Price Z-Score < -1.5, Stochastic < 15)
        # and quick price bounce confirmation (current price above SMA(5))
        if (current_rsi < 20 and
            current_z_score < -1.5 and
            current_stoch_k < 15 and
            current_last_price > sma_5):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_last_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))
        
        # --- Exit Conditions (SellSignal) ---
        # If (rsi > 60 or stoch_k > 80 or current_price < sma_5)
        if (current_rsi > 60 or 
            current_stoch_k > 80 or 
            current_last_price < sma_5):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_last_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))
            
    return signals