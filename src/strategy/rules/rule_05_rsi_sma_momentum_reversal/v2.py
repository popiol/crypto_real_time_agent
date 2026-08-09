from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

RULE_ID = "rsi_sma_filtered_reversal_v2"

# Define lookback periods
RSI_PERIOD = 14
SMA_PERIOD = 20
BB_PERIOD = 20
ZSCORE_PERIOD = 20

# Minimum candles required for all indicators to produce at least one value
# RSI(14) needs 14+2=16 candles to get current and previous RSI
# SMA(20) needs 20 candles
# BB(20) needs 20 candles
# Z-Score(20) needs 20 candles
MIN_CANDLES_FOR_ALL_INDICATORS = max(RSI_PERIOD + 2, SMA_PERIOD, BB_PERIOD, ZSCORE_PERIOD)

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
    # changes[i] = closes[i+1] - closes[i]
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]

    # Separate gains and losses
    gains = [max(0, change) for change in changes]
    losses = [abs(min(0, change)) for change in changes]

    avg_gains = []
    avg_losses = []
    rsi_values = []

    # Calculate initial average gain/loss over the first 'period' changes
    # This corresponds to candles[0]...candles[period] for the first period+1 candles
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
    # Loop starts from the (period+1)-th change, which corresponds to candle at index 'period+1'
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

def calculate_sma(candles: list[WarmCandle], period: int) -> list[float] | None:
    """
    Calculates the Simple Moving Average (SMA) for a list of WarmCandle objects.
    Returns a list of SMA values, where the last element is the current SMA.
    Returns None if insufficient data.
    """
    if len(candles) < period:
        return None

    closes = [c.close for c in candles]
    sma_values = []

    for i in range(len(closes) - period + 1):
        window = closes[i : i + period]
        sma_values.append(statistics.mean(window))
    
    return sma_values

def calculate_bollinger_bands(candles: list[WarmCandle], period: int, num_std_dev: float = 2.0) -> dict[str, list[float]] | None:
    """
    Calculates Bollinger Bands (Middle, Upper, Lower) for a list of WarmCandle objects.
    Returns a dictionary with lists of values for 'upper', 'middle', 'lower' bands.
    Returns None if insufficient data.
    """
    if len(candles) < period:
        return None

    closes = [c.close for c in candles]
    
    middle_band = calculate_sma(candles, period)
    if middle_band is None: # Should not happen if len(candles) >= period
        return None

    upper_band = []
    lower_band = []

    for i in range(len(closes) - period + 1):
        window = closes[i : i + period]
        
        # Calculate standard deviation for the current window
        # statistics.stdev requires at least 2 data points.
        if len(window) < 2:
            std_dev = 0.0
        else:
            std_dev = statistics.stdev(window)
        
        current_middle = middle_band[i]
        
        upper_band.append(current_middle + (std_dev * num_std_dev))
        lower_band.append(current_middle - (std_dev * num_std_dev))

    return {
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band
    }

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
        # If no volatility, Z-score is 0 as per pseudocode.
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

        # --- Calculate Indicators ---
        
        # RSI(14)
        rsi_values = calculate_rsi(warm_candles, RSI_PERIOD)
        if rsi_values is None or len(rsi_values) < 2: # Need current and previous RSI
            continue
        current_rsi = rsi_values[-1]
        previous_rsi = rsi_values[-2]

        # SMA(20)
        sma_20_values = calculate_sma(warm_candles, SMA_PERIOD)
        if sma_20_values is None or not sma_20_values:
            continue
        current_sma_20 = sma_20_values[-1]

        # Bollinger Bands (20)
        bb_bands = calculate_bollinger_bands(warm_candles, BB_PERIOD)
        if bb_bands is None or not bb_bands['middle']: # Ensure middle band (SMA) is calculated
            continue
        
        current_bb_upper = bb_bands['upper'][-1]
        current_bb_lower = bb_bands['lower'][-1]
        current_bb_middle = bb_bands['middle'][-1]

        bb_width = 0.0
        if current_bb_middle > 0:
            bb_width = (current_bb_upper - current_bb_lower) / current_bb_middle
        
        # Price Z-Score (20)
        # We need the last ZSCORE_PERIOD closes from warm_candles for context.
        # The current price (last_price from hot_ticks) is what we are scoring.
        recent_closes_for_zscore = [c.close for c in warm_candles] 
        price_z_score_current = calculate_price_z_score(recent_closes_for_zscore, ZSCORE_PERIOD, last_price)
        if price_z_score_current is None: # Should not happen if len(warm_candles) >= ZSCORE_PERIOD
            continue

        # Bid-Ask Spread Percentage
        spread_percentage = 0.0
        if last_price > 0:
            spread_percentage = ((current_ask - current_bid) / last_price) * 100
        
        # --- BUY Signal Logic ---
        buy_condition_rsi_cross = previous_rsi < 40 and current_rsi >= 40 # RSI crosses above 40
        buy_condition_price_above_sma = last_price > current_sma_20
        buy_condition_low_spread = spread_percentage < 0.3
        buy_condition_moderate_bbw = bb_width >= 0.015 and bb_width <= 0.030
        buy_condition_price_dip_zscore = price_z_score_current < -0.5

        if (buy_condition_rsi_cross and
            buy_condition_price_above_sma and
            buy_condition_low_spread and
            buy_condition_moderate_bbw and
            buy_condition_price_dip_zscore):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=last_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence as conditions are met
            ))
        
        # --- SELL Signal Logic (to close an existing long position) ---
        sell_condition_rsi_cross_below_60 = previous_rsi > 60 and current_rsi <= 60 # RSI crosses below 60
        sell_condition_price_below_sma = last_price < current_sma_20
        sell_condition_high_spread = spread_percentage > 0.5

        if (sell_condition_rsi_cross_below_60 or
            sell_condition_price_below_sma or
            sell_condition_high_spread):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=last_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence as conditions are met
            ))
            
    return signals