from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from pydantic import Field

RULE_ID = "momentum_confirmed_rebound_v2"

# Define lookback periods for indicators
RSI_PERIOD = 14
ZSCORE_PERIOD = 20
BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0 # Standard 2 standard deviations for Bollinger Bands
SMA_5_PERIOD = 5
SMA_20_PERIOD = 20

# Buy/Sell thresholds
RSI_OVERSOLD_BUY = 30
ZSCORE_OVERSOLD_BUY = -1.5
BB_WIDTH_THRESHOLD_BUY = 0.025 # Adjusted from 0.015 to 0.025

RSI_EXIT_SELL = 50
ZSCORE_EXIT_SELL = 0.5

# Minimum candles required for all indicators to produce at least two values for crossover (e.g., prev/current SMA, prev/current RSI).
# RSI(14) needs 14 + 2 = 16 candles for prev_rsi and current_rsi.
# Z-Score(20) needs 20 candles.
# BB Width(20) needs 20 candles.
# SMA(20) needs 20 candles for the first value, and 21 for prev_sma_20 and current_sma_20.
MIN_CANDLES_FOR_ALL_INDICATORS = max(RSI_PERIOD + 2, ZSCORE_PERIOD, BB_PERIOD, SMA_20_PERIOD + 1)

def calculate_rsi(candles: list[WarmCandle], period: int) -> list[float] | None:
    """
    Calculates the Relative Strength Index (RSI) for a list of WarmCandle objects.
    Returns a list of RSI values.
    Returns None if insufficient data.
    """
    # Need at least period + 1 candles for the first RSI value.
    # To get prev_rsi and current_rsi, we need period + 2 candles.
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

def calculate_sma(closes: list[float], period: int) -> list[float] | None:
    """
    Calculates the Simple Moving Average (SMA) for a list of closing prices.
    Returns a list of SMA values, where each value corresponds to the end of a period.
    Returns None if insufficient data.
    """
    if len(closes) < period:
        return None

    sma_values = []
    for i in range(period - 1, len(closes)):
        sma_values.append(statistics.mean(closes[i - period + 1 : i + 1]))
    return sma_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Ensure enough warm candles for all indicators, including prev/current values
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
        if rsi_values is None or len(rsi_values) < 2: # Need at least two RSI values for prev/current check
            continue
        current_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2]

        # Price Z-Score (20) - using the latest tick price for responsiveness
        price_z_score_current = calculate_price_z_score(warm_closes, ZSCORE_PERIOD, last_price)
        if price_z_score_current is None:
            continue

        # Bollinger Band Width (20)
        current_bb_width = calculate_bollinger_band_width(warm_candles, BB_PERIOD, BB_STD_DEV_MULTIPLIER)
        if current_bb_width is None:
            continue
        
        # SMAs for crossover
        sma_5_values = calculate_sma(warm_closes, SMA_5_PERIOD)
        sma_20_values = calculate_sma(warm_closes, SMA_20_PERIOD)

        # Check for sufficient SMA values for crossover.
        # sma_20_values will be the shortest list for the lookback periods used.
        # We need at least 2 values in sma_20_values for prev/current comparison.
        if sma_5_values is None or sma_20_values is None or len(sma_5_values) < 2 or len(sma_20_values) < 2:
            continue
        
        # Align the SMA values: the last two values for each SMA correspond to the last two candles.
        current_sma_5 = sma_5_values[-1]
        prev_sma_5 = sma_5_values[-2]
        current_sma_20 = sma_20_values[-1]
        prev_sma_20 = sma_20_values[-2]

        # --- BUY Signal Conditions ---
        # 1. RSI(14) is oversold (< 30)
        # 2. Price Z-Score(20) is significantly undervalued (< -1.5)
        # 3. SMA(5) crosses above SMA(20) (momentum confirmation)
        # 4. Bollinger Band Width(20) is below 0.025 (relaxed volatility condition)
        buy_condition = (
            (current_rsi < RSI_OVERSOLD_BUY) and
            (price_z_score_current < ZSCORE_OVERSOLD_BUY) and
            (current_sma_5 > current_sma_20) and
            (prev_sma_5 <= prev_sma_20) and # Crossover confirmation
            (current_bb_width < BB_WIDTH_THRESHOLD_BUY)
        )

        if buy_condition:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=last_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence as strict conditions are met
            ))
        
        # --- SELL Signal Conditions (to close an existing long position) ---
        # 1. RSI(14) rises above 50 (no longer oversold, potentially overbought)
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