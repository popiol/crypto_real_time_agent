from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

RULE_ID = "rsi_sma_reversal_v1"
RSI_PERIOD = 14
SMA_PERIOD = 5
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
# Need RSI_PERIOD + 2 candles to get current and previous RSI values
# (period for initial average, +1 for first RSI, +1 for previous RSI)
MIN_CANDLES_FOR_RSI = RSI_PERIOD + 2

def calculate_rsi(candles: list[WarmCandle], period: int) -> list[float] | None:
    """
    Calculates the Relative Strength Index (RSI) for a list of WarmCandle objects.
    Returns a list of RSI values, where the last two elements are the current and
    previous RSI, respectively. Returns None if insufficient data.
    """
    if len(candles) < period + 1: # Need at least period+1 candles for the first RSI value
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

    # Calculate the first RSI value
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

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for RSI and SMA calculations
        if len(pair_data.warm) < MIN_CANDLES_FOR_RSI:
            continue

        # Ensure we have hot data for current real-time price and timestamp
        if not pair_data.hot:
            continue
        
        current_tick = pair_data.hot[-1]
        current_tick_price = current_tick.last_price
        timestamp = current_tick.polled_at

        # --- Calculate Indicators ---
        
        # RSI(14)
        rsi_values = calculate_rsi(pair_data.warm, RSI_PERIOD)
        if rsi_values is None or len(rsi_values) < 2: # Need current and previous RSI
            continue
        current_rsi = rsi_values[-1]
        previous_rsi = rsi_values[-2]

        # SMA(5)
        sma_values = calculate_sma(pair_data.warm, SMA_PERIOD)
        if sma_values is None or not sma_values: # Need current SMA
            continue
        current_sma = sma_values[-1]

        # Get the close price of the last *completed* hourly candle for SMA cross comparison
        last_closed_candle_price = pair_data.warm[-1].close

        # --- BUY Signal Logic ---
        # 1. RSI(14) crosses above 30 (from below or equal)
        # 2. Current real-time price is above the 5-period SMA
        if (previous_rsi <= RSI_OVERSOLD and current_rsi > RSI_OVERSOLD and
                current_tick_price > current_sma):
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=current_tick_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence as conditions are met
            ))
        
        # --- SELL Signal Logic (to close an existing long position) ---
        # 1. RSI(14) crosses below 70 (from above or equal)
        # OR
        # 2. Last completed hourly candle's close was above SMA(5)
        #    AND current real-time price is below SMA(5) (price cross below SMA)
        elif (previous_rsi >= RSI_OVERBOUGHT and current_rsi < RSI_OVERBOUGHT) or \
             (last_closed_candle_price >= current_sma and current_tick_price < current_sma):
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=current_tick_price,
                rule_id=RULE_ID,
                confidence=1.0 # High confidence as conditions are met
            ))
            
    return signals