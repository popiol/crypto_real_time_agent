from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

RULE_ID = "oversold-reversal-fix-v2"

# Define lookback periods for indicators
RSI_PERIOD = 14
STOCH_K_PERIOD = 14
Z_SCORE_PERIOD = 20
SMA_PERIOD = 5

# Minimum candles required for all indicators to produce at least one value.
# RSI(14) needs 14+1 = 15 candles.
# Stochastic K(14) needs 14 candles.
# Z-Score(20) needs 20 closes.
# SMA(5) needs 5 closes.
MIN_CANDLES_REQUIRED = max(RSI_PERIOD + 1, STOCH_K_PERIOD, Z_SCORE_PERIOD, SMA_PERIOD)


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


def calculate_price_z_score(candles: list[WarmCandle], period: int) -> float | None:
    """
    Calculates the Z-Score of the current_price relative to the mean and standard deviation
    of the recent 'period' closing prices.
    Returns None if insufficient data.
    """
    if len(candles) < period:
        return None
    
    recent_closes = [c.close for c in candles[-period:]]
    current_price = candles[-1].close

    if not recent_closes: 
        return None

    mean_closes = statistics.mean(recent_closes)
    
    if len(recent_closes) > 1:
        std_dev_closes = statistics.stdev(recent_closes)
    else:
        std_dev_closes = 0.0

    if std_dev_closes == 0:
        return 0.0 if current_price == mean_closes else (1.0 if current_price > mean_closes else -1.0)
    else:
        return (current_price - mean_closes) / std_dev_closes


def calculate_stochastic_oscillator(candles: list[WarmCandle], k_period: int) -> float | None:
    """
    Calculates the Stochastic Oscillator %K for the last candle.
    Returns the last %K value.
    Returns None if insufficient data.
    """
    if len(candles) < k_period:
        return None

    period_candles = candles[-k_period:]
    
    highest_high = max(c.high for c in period_candles)
    lowest_low = min(c.low for c in period_candles)
    current_close = period_candles[-1].close

    if (highest_high - lowest_low) == 0:
        k = 0.0 if current_close == lowest_low else (100.0 if current_close == highest_high else 50.0)
    else:
        k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
    
    return k


def calculate_sma(candles: list[WarmCandle], period: int) -> float | None:
    """
    Calculates the Simple Moving Average (SMA) for the last 'period' closing prices.
    Returns the last SMA value.
    Returns None if insufficient data.
    """
    if len(candles) < period:
        return None
    
    closes = [c.close for c in candles[-period:]]
    
    if not closes:
        return None
        
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

        # --- Calculate Indicators ---
        
        current_rsi = calculate_rsi(warm_candles, RSI_PERIOD)
        current_stoch_k = calculate_stochastic_oscillator(warm_candles, STOCH_K_PERIOD)
        current_price_z_score = calculate_price_z_score(warm_candles, Z_SCORE_PERIOD)
        current_sma_5 = calculate_sma(warm_candles, SMA_PERIOD)

        # Skip if any indicator could not be calculated
        if any(v is None for v in [current_rsi, current_stoch_k, current_price_z_score, current_sma_5]):
            continue

        # --- Entry Conditions (BuySignal) ---
        # 1. Broadened 'deeply oversold' definition:
        #    a) RSI < 25 AND Price Z-Score < -2.0
        #    OR
        #    b) Stochastic Oscillator %K < 10
        condition_rsi_zscore = (current_rsi < 25) and (current_price_z_score < -2.0)
        condition_stoch = (current_stoch_k < 10)
        
        # 2. Simplified price action confirmation: current_price > SMA(5)
        price_confirmation = (current_last_price > current_sma_5)

        if (condition_rsi_zscore or condition_stoch) and price_confirmation:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_last_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence for reversal
            ))
        
        # --- Exit Conditions (SellSignal) ---
        # Close existing long position when:
        # 1. RSI moves into overbought territory (RSI > 60)
        # OR
        # 2. Price breaches below the 5-period SMA, indicating loss of momentum.
        if (current_rsi > 60) or (current_last_price < current_sma_5):
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_last_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence for exiting a long
            ))
            
    return signals