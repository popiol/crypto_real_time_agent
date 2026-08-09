from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

RULE_ID = "fix_deep_value_reversal_v2"

# Define lookback periods for indicators
RSI_PERIOD = 14
SMA_PERIOD = 23
ZSCORE_PERIOD = 23

# Minimum candles required for all indicators to produce at least one value.
# RSI(14) needs 14+2=16 candles to get current and previous RSI.
# SMA(23) needs 23 candles.
# Z-Score(23) needs 23 candles.
MIN_CANDLES_FOR_ALL_INDICATORS = max(RSI_PERIOD + 2, SMA_PERIOD, ZSCORE_PERIOD)

def calculate_rsi(candles: list[WarmCandle], period: int) -> list[float] | None:
    """
    Calculates the Relative Strength Index (RSI) for a list of WarmCandle objects.
    Returns a list of RSI values. The last two elements are the current and
    previous RSI, respectively, if enough data is available.
    Returns None if insufficient data.
    """
    # Need at least period + 1 candles for the first RSI value,
    # and period + 2 for current and previous RSI values.
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

def calculate_sma(candles: list[WarmCandle], period: int) -> float | None:
    """
    Calculates the Simple Moving Average (SMA) for a list of WarmCandle objects.
    Returns the current SMA value.
    Returns None if insufficient data.
    """
    if len(candles) < period:
        return None

    closes = [c.close for c in candles]
    
    # Calculate SMA for the last 'period' closes
    window = closes[-period:]
    return statistics.mean(window)

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

    if not recent_closes: # Should not happen if len(closes) >= period
        return None

    mean_closes = statistics.mean(recent_closes)
    
    # Handle case where all prices are the same (no volatility)
    if len(recent_closes) > 1:
        std_dev_closes = statistics.stdev(recent_closes)
    else: # If only one data point, standard deviation is 0
        std_dev_closes = 0.0

    if std_dev_closes == 0:
        # If no volatility, Z-score is 0.0 as per common interpretation, or undefined.
        # For trading, 0.0 implies current_price is at the mean.
        return 0.0
    else:
        return (current_price - mean_closes) / std_dev_closes

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
        
        # SMA(23)
        current_sma_23 = calculate_sma(warm_candles, SMA_PERIOD)
        if current_sma_23 is None:
            continue

        # RSI(14)
        rsi_values = calculate_rsi(warm_candles, RSI_PERIOD)
        if rsi_values is None or len(rsi_values) < 2: # Need current and previous RSI
            continue
        current_rsi = rsi_values[-1]
        # previous_rsi = rsi_values[-2] # Not directly used in the new logic, but kept for context if needed

        # Price Z-Score (23)
        price_z_score_current = calculate_price_z_score(warm_closes, ZSCORE_PERIOD, last_price)
        if price_z_score_current is None:
            continue

        # Bid-Ask Spread Percentage
        bid_ask_spread_percent = 0.0
        if current_ask > 0: # Avoid division by zero
            bid_ask_spread_percent = ((current_ask - current_bid) / current_ask) * 100
        
        # --- BUY Signal Logic ---
        # IF current_price > SMA_23
        # AND RSI_14 >= 25 AND RSI_14 <= 35
        # AND Z_Score_23 < -0.75
        # AND bid_ask_spread_percent < 0.5
        
        buy_condition_trend = last_price > current_sma_23
        buy_condition_oversold_rsi = current_rsi >= 25 and current_rsi <= 35
        buy_condition_deep_value_zscore = price_z_score_current < -0.75
        buy_condition_liquid_entry = bid_ask_spread_percent < 0.5

        if (buy_condition_trend and
            buy_condition_oversold_rsi and
            buy_condition_deep_value_zscore and
            buy_condition_liquid_entry):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=last_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence as conditions are met
            ))
        
        # --- SELL Signal Logic (to close an existing long position) ---
        # IF RSI_14 > 60 OR current_price < SMA_23

        sell_condition_momentum_faded = current_rsi > 60
        sell_condition_trend_broken = last_price < current_sma_23

        if (sell_condition_momentum_faded or
            sell_condition_trend_broken):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=last_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence as conditions are met
            ))
            
    return signals