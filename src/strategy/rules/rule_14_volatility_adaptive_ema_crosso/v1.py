from __future__ import annotations
import numpy as np
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Constants ---
EMA_SHORT_PERIOD = 10
EMA_LONG_PERIOD = 30
ATR_PERIOD = 14
K_ATR = 1.0  # Constant for volatility threshold multiplier
N_ATR_HISTORY = 20  # Historical window for average ATR (number of past ATR values)
M_DEVIATION = 0.5  # Multiplier for deviation from EMA_long during high volatility

# Minimum number of warm candles required for all calculations:
# 1. For EMA_LONG_PERIOD: Need at least EMA_LONG_PERIOD candles.
# 2. For ATR and its historical average:
#    - To calculate the first ATR value, we need ATR_PERIOD candles.
#    - To get N_ATR_HISTORY valid ATR values, we need ATR_PERIOD + (N_ATR_HISTORY - 1) candles.
#    Example: if ATR_PERIOD=14, N_ATR_HISTORY=20. We need 14 candles for the first ATR.
#             Then 19 more candles to have 20 ATR values. Total: 14 + 19 = 33 candles.
# So, MIN_CANDLES = max(EMA_LONG_PERIOD, ATR_PERIOD + N_ATR_HISTORY - 1)
MIN_CANDLES = max(EMA_LONG_PERIOD, ATR_PERIOD + N_ATR_HISTORY - 1)

# --- Helper Functions ---

def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates Exponential Moving Average using numpy."""
    if len(prices) < period:
        return np.full_like(prices, np.nan) # Return NaN array if not enough data

    ema = np.zeros_like(prices, dtype=float)
    alpha = 2 / (period + 1)

    # Initialize the first EMA value with the Simple Moving Average of the first 'period' prices
    ema[period - 1] = np.mean(prices[:period])

    # Calculate subsequent EMA values iteratively
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    
    # Fill leading NaNs for periods where EMA is not yet defined
    ema[:period - 1] = np.nan 
    return ema

def calculate_tr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Calculates True Range (TR) values."""
    tr = np.zeros_like(closes, dtype=float)
    # The first True Range is simply High - Low
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    return tr

def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Calculates Average True Range (ATR)."""
    if len(highs) < period:
        return np.full_like(highs, np.nan) # Return NaN array if not enough data

    tr_values = calculate_tr(highs, lows, closes)
    atr = np.zeros_like(tr_values, dtype=float)

    # Initialize the first ATR value with the Simple Moving Average of the first 'period' TR values
    atr[period - 1] = np.mean(tr_values[:period])

    # Calculate subsequent ATR values iteratively
    for i in range(period, len(tr_values)):
        atr[i] = (atr[i-1] * (period - 1) + tr_values[i]) / period
    
    # Fill leading NaNs for periods where ATR is not yet defined
    atr[:period - 1] = np.nan
    return atr

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates buy and sell signals based on a Volatility-Adaptive EMA Crossover strategy.
    
    This rule generates buy and sell signals based on the crossover of two Exponential Moving Averages (EMAs),
    dynamically adjusting the signal sensitivity or applying a filter based on the current market volatility,
    measured by Average True Range (ATR). During periods of high volatility, the rule requires a stronger
    confirmation (e.g., price significantly deviating from EMAs) to reduce whipsaws.
    """
    signals: list[BuySignal | SellSignal] = []
    rule_id = "cad9c41f-3c55-4386-b022-9b4b627dd08c"

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        
        # Ensure we have enough historical data (warm candles) for calculations
        if len(warm_candles) < MIN_CANDLES:
            continue

        # Extract necessary data from warm candles.
        # warm_candles are ordered oldest to newest.
        closes = np.array([c.close for c in warm_candles], dtype=float)
        highs = np.array([c.high for c in warm_candles], dtype=float)
        lows = np.array([c.low for c in warm_candles], dtype=float)

        # 1. Calculate EMAs
        ema_short = calculate_ema(closes, EMA_SHORT_PERIOD)
        ema_long = calculate_ema(closes, EMA_LONG_PERIOD)

        # Check if EMA calculations resulted in valid arrays for the latest data points
        if np.isnan(ema_short[-1]) or np.isnan(ema_long[-1]):
            continue

        # 2. Calculate ATR
        atr_values = calculate_atr(highs, lows, closes, ATR_PERIOD)
        if np.isnan(atr_values[-1]):
            continue

        # Get current and previous indicator values for signal generation
        current_close = closes[-1]
        current_timestamp = warm_candles[-1].hour

        # EMA values for current (t) and previous (t-1) periods
        prev_ema_short = ema_short[-2]
        current_ema_short = ema_short[-1]
        prev_ema_long = ema_long[-2]
        current_ema_long = ema_long[-1]
        
        # Current ATR value
        current_atr = atr_values[-1]

        # 3. Define volatility threshold
        # We need N_ATR_HISTORY of the most recent valid ATR values for the average.
        # MIN_CANDLES ensures that `atr_values` contains at least `N_ATR_HISTORY` valid entries at its tail.
        historical_atr_for_threshold = atr_values[-N_ATR_HISTORY:]
        
        # Filter out any potential NaNs if MIN_CANDLES calculation was tight or data was sparse
        historical_atr_for_threshold = historical_atr_for_threshold[~np.isnan(historical_atr_for_threshold)]
        
        if len(historical_atr_for_threshold) == 0:
            continue # Not enough valid ATR history to determine threshold

        average_historical_atr = np.mean(historical_atr_for_threshold)
        volatility_threshold = K_ATR * average_historical_atr

        # 4. Determine if the current market is in a 'high volatility' regime
        is_high_volatility = current_atr > volatility_threshold

        # 5. Generate Buy Signal
        # EMA_short crosses above EMA_long
        buy_crossover = (prev_ema_short <= prev_ema_long) and (current_ema_short > current_ema_long)
        
        if buy_crossover:
            # Apply volatility-adaptive filter
            if not is_high_volatility:
                # Low volatility: simple EMA crossover is sufficient
                signals.append(BuySignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_close,
                    rule_id=rule_id,
                    confidence=1.0 # Default confidence
                ))
            elif is_high_volatility and (current_close > current_ema_long + M_DEVIATION * current_atr):
                # High volatility: requires stronger confirmation (price significantly above EMA_long)
                signals.append(BuySignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_close,
                    rule_id=rule_id,
                    confidence=1.0
                ))

        # 6. Generate Sell Signal
        # EMA_short crosses below EMA_long
        sell_crossover = (prev_ema_short >= prev_ema_long) and (current_ema_short < current_ema_long)

        if sell_crossover:
            # Apply volatility-adaptive filter
            if not is_high_volatility:
                # Low volatility: simple EMA crossover is sufficient
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_close,
                    rule_id=rule_id,
                    confidence=1.0
                ))
            elif is_high_volatility and (current_close < current_ema_long - M_DEVIATION * current_atr):
                # High volatility: requires stronger confirmation (price significantly below EMA_long)
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_close,
                    rule_id=rule_id,
                    confidence=1.0
                ))

    return signals