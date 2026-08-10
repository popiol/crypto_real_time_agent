from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from pydantic import Field

RULE_ID = "oversold_reversal_momentum_confirm"

# Define lookback periods for indicators
RSI_PERIOD = 14
Z_SCORE_PERIOD = 20
BBW_PERIOD = 20 # Bollinger Band Width period
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3 # %D is not used in rule logic, but kept for consistency with common Stochastic implementations

# Minimum candles required for all indicators to produce at least one value,
# including 'previous' values for momentum confirmation.
# RSI(14) needs 14+1 = 15 candles for one value. To get prev_rsi, we need 15 candles for warm_candles[:-1], so total 16.
# Z-Score(20) needs 20 closes.
# Stochastic K(14) needs 14 candles. To get prev_stoch_k, we need 14 candles for warm_candles[:-1], so total 15.
# BBW(20) needs 20 closes.
# The maximum lookback requirement is 20 candles for Z-Score and BBW.
# This means we need at least 20 candles for the current values of Z-Score and BBW.
# If len(warm_candles) is 20:
#   - current_rsi will be calculated from 20 candles (enough for 15 required)
#   - prev_rsi will be calculated from 19 candles (enough for 15 required)
#   - current_z_score will be calculated from 20 closes (enough for 20 required)
#   - current_stoch_k will be calculated from 20 candles (enough for 14 required)
#   - prev_stoch_k will be calculated from 19 candles (enough for 14 required)
#   - current_bb_width will be calculated from 20 closes (enough for 20 required)
MIN_CANDLES_REQUIRED = max(
    RSI_PERIOD + 1 + 1, # For prev_rsi: (period + 1) for a value, +1 for previous candle
    Z_SCORE_PERIOD,
    STOCH_K_PERIOD + 1, # For prev_stoch_k: period for a value, +1 for previous candle
    BBW_PERIOD
)
# Recalculating:
# RSI needs 15 for current. prev_rsi needs 15 for warm_candles[:-1]. So len(warm_candles)-1 >= 15 => len(warm_candles) >= 16.
# Stoch K needs 14 for current. prev_stoch_k needs 14 for warm_candles[:-1]. So len(warm_candles)-1 >= 14 => len(warm_candles) >= 15.
# Z-Score needs 20.
# BBW needs 20.
# So, min_candles = max(16, 20, 15, 20) = 20.
MIN_CANDLES_REQUIRED = 20


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
    # The loop should start from 'period' index in 'changes' list
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


def calculate_bollinger_band_width(closes: list[float], period: int, num_std_dev: float = 2.0) -> float | None:
    """
    Calculates the Bollinger Band Width for a list of closing prices.
    BBW = (Upper Band - Lower Band) / Middle Band (SMA)
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

    # BBW = (Upper Band - Lower Band) / SMA = ( (SMA + N*std_dev) - (SMA - N*std_dev) ) / SMA
    # BBW = (2 * N * std_dev) / SMA
    
    if sma == 0: # Avoid division by zero, especially if prices are all zero (highly unlikely)
        return 0.0 
    
    return (2 * std_dev * num_std_dev) / sma


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Ensure enough warm candles for all indicators, including previous values
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
        
        # Previous RSI(14) for momentum confirmation
        prev_rsi = calculate_rsi(warm_candles[:-1], RSI_PERIOD)
        # prev_rsi can be None if warm_candles[:-1] is exactly period+1-1 = period candles long,
        # but current_rsi needs period+1. Our MIN_CANDLES_REQUIRED handles this.
        # If MIN_CANDLES_REQUIRED is 20, warm_candles[:-1] has 19 candles, which is enough for RSI(14) (needs 15).
        if prev_rsi is None:
             continue # Defensive check, should not be hit with correct MIN_CANDLES_REQUIRED

        # Price Z-Score (20) - using the last warm candle's close for the calculation base
        current_z_score = calculate_price_z_score(warm_closes, Z_SCORE_PERIOD, warm_candles[-1].close)
        if current_z_score is None:
            continue

        # Current Stochastic Oscillator %K (14)
        current_stoch_k = calculate_stochastic_oscillator(warm_candles, STOCH_K_PERIOD, STOCH_D_PERIOD)
        if current_stoch_k is None:
            continue

        # Previous Stochastic Oscillator %K (14) for momentum confirmation
        prev_stoch_k = calculate_stochastic_oscillator(warm_candles[:-1], STOCH_K_PERIOD, STOCH_D_PERIOD)
        # If MIN_CANDLES_REQUIRED is 20, warm_candles[:-1] has 19 candles, which is enough for Stoch K(14) (needs 14).
        if prev_stoch_k is None:
            continue # Defensive check

        # Bollinger Band Width (20)
        current_bb_width = calculate_bollinger_band_width(warm_closes, BBW_PERIOD)
        if current_bb_width is None:
            continue

        # --- Entry Conditions (BuySignal) ---
        # Relaxed Oversold with Momentum Confirmation and Low Volatility Context
        if (current_rsi < 35 and
            current_rsi > prev_rsi and # RSI turning up
            current_stoch_k < 20 and
            current_stoch_k > prev_stoch_k and # Stochastic K turning up
            current_z_score < -1.5 and
            current_bb_width < 0.015):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_last_price,
                rule_id=RULE_ID,
                confidence=1.0 # Default confidence
            ))
        
        # --- Exit Conditions (SellSignal) ---
        # Overbought RSI, signaling a potential peak for the bounce
        if (current_rsi > 60):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_last_price,
                rule_id=RULE_ID,
                confidence=1.0 # Default confidence
            ))
            
    return signals