from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from pydantic import Field

RULE_ID = "momentum_confirmed_oversold_v1"

# Define lookback periods for indicators
RSI_PERIOD = 14
ZSCORE_PERIOD = 20
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
SMA_PERIOD = 5

# Minimum candles required for all indicators to produce at least one value.
# RSI(14) needs 14 + 1 = 15 candles for the first RSI value.
# Z-Score(20) needs 20 candles.
# BB Width(20) needs 20 candles.
# Stochastic K(14) needs 14 candles. Stochastic D(3) then needs 3 values of K, so 14 + (3-1) = 16 candles for the first D value.
# SMA(5) needs 5 candles.
# The maximum lookback requirement is 20 candles.
MIN_CANDLES_FOR_ALL_INDICATORS = max(
    RSI_PERIOD + 1,
    ZSCORE_PERIOD,
    BB_PERIOD,
    STOCH_K_PERIOD + STOCH_D_PERIOD - 1,
    SMA_PERIOD
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


def calculate_sma(closes: list[float], period: int) -> list[float] | None:
    """
    Calculates the Simple Moving Average (SMA) for a list of closing prices.
    Returns a list of SMA values.
    Returns None if insufficient data.
    """
    if len(closes) < period:
        return None
    
    sma_values = []
    for i in range(period - 1, len(closes)):
        sma = statistics.mean(closes[i - period + 1 : i + 1])
        sma_values.append(sma)
        
    return sma_values


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

        # We need at least 2 candles for previous_close and current_close comparison,
        # and for SMA cross-over logic.
        if len(warm_candles) < 2:
            continue

        current_close = warm_candles[-1].close
        previous_close = warm_candles[-2].close

        # --- Prepare data for indicators ---
        warm_closes = [c.close for c in warm_candles]

        # --- Calculate Indicators ---
        
        # RSI(14)
        rsi_values = calculate_rsi(warm_candles, RSI_PERIOD)
        if rsi_values is None or not rsi_values:
            continue
        current_rsi = rsi_values[-1]

        # Price Z-Score (20) - using the latest tick price for responsiveness
        price_z_score_current = calculate_price_z_score(warm_closes, ZSCORE_PERIOD, current_price)
        if price_z_score_current is None:
            continue

        # Bollinger Band Width (20)
        current_bb_width = calculate_bollinger_band_width(warm_candles, BB_PERIOD, BB_STD_DEV_MULTIPLIER)
        if current_bb_width is None:
            continue

        # Stochastic Oscillator (K=14, D=3)
        stoch_k_values, stoch_d_values = calculate_stochastic_oscillator(warm_candles, STOCH_K_PERIOD, STOCH_D_PERIOD)
        if stoch_k_values is None or not stoch_k_values:
            continue
        current_stoch_k = stoch_k_values[-1]

        # SMA(5)
        sma_5_values = calculate_sma(warm_closes, SMA_PERIOD)
        if sma_5_values is None or len(sma_5_values) < 2: # Need current and previous SMA for cross-over
            continue
        current_sma_5 = sma_5_values[-1]
        # The previous SMA_5 corresponds to the candle before the last one.
        # This is for comparing the close of warm_candles[-2] against sma_5_values[-2]
        previous_sma_5 = sma_5_values[-2]

        # --- BUY Signal Conditions (Momentum-Confirmed Oversold Reversal) ---
        buy_signal_triggered = False

        # Condition Set 1: Low Volatility Reversal
        # BB_Width < 0.01 AND RSI < 25 AND Z_Score < -1.5 AND Stoch_K < 15
        if (current_bb_width < 0.01 and
            current_rsi < 25 and
            price_z_score_current < -1.5 and
            current_stoch_k < 15):
            
            # Momentum Confirmation: Price closes above SMA_5, having been at or below it previously
            # We use current_close against current_sma_5 and previous_close against previous_sma_5
            if current_close > current_sma_5 and previous_close <= previous_sma_5:
                buy_signal_triggered = True

        # Condition Set 2: Moderate/High Volatility Reversal
        # BB_Width > 0.1 AND RSI < 40 AND Z_Score < -2.0 AND Stoch_K < 15
        elif (current_bb_width > 0.1 and
              current_rsi < 40 and
              price_z_score_current < -2.0 and
              current_stoch_k < 15):
            
            # Momentum Confirmation: Price closes above SMA_5, having been at or below it previously
            # We use current_close against current_sma_5 and previous_close against previous_sma_5
            if current_close > current_sma_5 and previous_close <= previous_sma_5:
                buy_signal_triggered = True

        if buy_signal_triggered:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))
        
        # --- SELL Signal Conditions (Exit existing long position) ---
        # Price crosses below SMA_5 from above
        # current_close < SMA_5 AND previous_close >= SMA_5 (using candle closes and SMAs)
        sell_condition = (current_close < current_sma_5 and previous_close >= previous_sma_5)

        if sell_condition:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_price,
                rule_id=RULE_ID,
                confidence=1.0
            ))
            
    return signals