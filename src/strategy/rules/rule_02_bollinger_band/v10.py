from __future__ import annotations

import numpy as np
from datetime import datetime, timedelta
import statistics

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle


# --- Parameters for Bollinger Bands ---
BB_PERIOD = 20  # Number of candles for Bollinger Band calculation (e.g., 20 for 20-period SMA)
BB_STD_DEV = 2.0  # Standard deviation multiplier for Bollinger Bands (e.g., 2 standard deviations)

# --- Parameters for RSI ---
RSI_PERIOD = 14  # Number of candles for RSI calculation (e.g., 14 periods)
RSI_OVERSOLD_THRESHOLD = 30  # RSI level below which an asset is considered oversold
RSI_OVERBOUGHT_THRESHOLD = 70  # RSI level above which an asset is considered overbought

# --- Parameters for ADX Trend Filter ---
HIGHER_TIMEFRAME_HOURS = 4  # Higher timeframe for ADX (e.g., 4h)
ADX_PERIOD = 14  # Number of periods for ADX calculation
ADX_TREND_THRESHOLD = 25.0  # ADX value below which market is considered weak/non-trending

# Minimum candles required for primary timeframe (1h) calculations:
# BB_PERIOD candles for BB's SMA and StdDev.
# RSI_PERIOD + 1 candles for RSI (to get RSI_PERIOD price changes).
MIN_PRIMARY_TIMEFRAME_CANDLES = max(BB_PERIOD, RSI_PERIOD + 1)

# Minimum candles required for higher timeframe (HTF) ADX calculation:
# A robust ADX calculation needs at least 2 * ADX_PERIOD candles.
MIN_HTF_CANDLES_FOR_ADX = 2 * ADX_PERIOD

# Total 1h candles needed for both primary timeframe indicators and HTF ADX:
# Take the maximum of primary timeframe needs and (HTF candles needed * hours per HTF candle).
MIN_CANDLES_REQUIRED_TOTAL = max(
    MIN_PRIMARY_TIMEFRAME_CANDLES,
    MIN_HTF_CANDLES_FOR_ADX * HIGHER_TIMEFRAME_HOURS
)


def calculate_rsi(prices: np.ndarray, period: int) -> float:
    """
    Calculates the Relative Strength Index (RSI) for a given series of prices.
    Assumes prices are ordered from oldest to newest.
    
    Args:
        prices (np.ndarray): A numpy array of closing prices.
        period (int): The number of periods to use for RSI calculation.

    Returns:
        float: The latest RSI value, or np.nan if not enough data.
    """
    if len(prices) < period + 1:
        return np.nan

    diffs = np.diff(prices)
    gains = np.maximum(0, diffs)
    losses = np.maximum(0, -diffs)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rs = np.inf if avg_gain > 0 else 1.0 # If no losses, RSI is 100. If no movement, RSI is 50.
    else:
        rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    for i in range(period, len(gains)):
        current_gain = gains[i]
        current_loss = losses[i]

        avg_gain = ((avg_gain * (period - 1)) + current_gain) / period
        avg_loss = ((avg_loss * (period - 1)) + current_loss) / period

        if avg_loss == 0:
            rs = np.inf if avg_gain > 0 else 1.0
        else:
            rs = avg_gain / avg_loss
        
        rsi = 100 - (100 / (1 + rs))
            
    return rsi


def resample_candles(
    warm_candles: list[WarmCandle], target_interval_hours: int
) -> list[WarmCandle]:
    """
    Resamples a list of 1-hour WarmCandle objects into a higher timeframe.
    Assumes warm_candles are sorted by hour (oldest to newest).
    Only returns fully formed HTF candles.
    """
    if not warm_candles or target_interval_hours <= 1:
        return warm_candles

    resampled_candles: list[WarmCandle] = []
    
    grouped_candles: dict[datetime, list[WarmCandle]] = {}

    for candle in warm_candles:
        # Calculate the start of the HTF period for this candle
        htf_period_start = candle.hour.replace(minute=0, second=0, microsecond=0)
        htf_period_start -= timedelta(hours=htf_period_start.hour % target_interval_hours)
        
        if htf_period_start not in grouped_candles:
            grouped_candles[htf_period_start] = []
        grouped_candles[htf_period_start].append(candle)
    
    sorted_htf_starts = sorted(grouped_candles.keys())

    for htf_start in sorted_htf_starts:
        group = grouped_candles[htf_start]
        # Only form a new HTF candle if the group is complete
        if len(group) == target_interval_hours:
            open_p = group[0].open_price
            high_p = max(c.high for c in group)
            low_p = min(c.low for c in group)
            close_p = group[-1].close
            avg_spread = np.mean([c.avg_spread_rel for c in group])

            resampled_candles.append(
                WarmCandle(
                    hour=htf_start, # The start of the HTF period
                    open_price=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    avg_spread_rel=avg_spread,
                )
            )
    return resampled_candles


def calculate_adx(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
) -> float:
    """
    Calculates the Average Directional Index (ADX).
    Assumes arrays are ordered from oldest to newest.
    Returns the latest ADX value, or np.nan if not enough data.
    Minimum data required: 2 * period candles for a robust ADX calculation.
    """
    if len(highs) < 2 * period:
        return np.nan

    tr_values = np.zeros(len(highs))
    plus_dm = np.zeros(len(highs))
    minus_dm = np.zeros(len(highs))

    for i in range(1, len(highs)):
        # True Range
        tr_values[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )

        # Directional Movement
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]

        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # --- Wilder's Smoothing ---
    smoothed_tr = np.zeros(len(highs))
    smoothed_plus_dm = np.zeros(len(highs))
    smoothed_minus_dm = np.zeros(len(highs))

    # Calculate initial averages for the first 'period' values (indices 1 to period)
    smoothed_tr_prev = np.sum(tr_values[1 : period + 1]) / period
    smoothed_plus_dm_prev = np.sum(plus_dm[1 : period + 1]) / period
    smoothed_minus_dm_prev = np.sum(minus_dm[1 : period + 1])

    smoothed_tr[period] = smoothed_tr_prev
    smoothed_plus_dm[period] = smoothed_plus_dm_prev
    smoothed_minus_dm[period] = smoothed_minus_dm_prev

    # Apply smoothing for subsequent values
    for i in range(period + 1, len(highs)):
        smoothed_tr_prev = (smoothed_tr_prev * (period - 1) + tr_values[i]) / period
        smoothed_plus_dm_prev = (smoothed_plus_dm_prev * (period - 1) + plus_dm[i]) / period
        smoothed_minus_dm_prev = (smoothed_minus_dm_prev * (period - 1) + minus_dm[i]) / period
        
        smoothed_tr[i] = smoothed_tr_prev
        smoothed_plus_dm[i] = smoothed_plus_dm_prev
        smoothed_minus_dm[i] = smoothed_minus_dm_prev

    # --- Calculate DI+, DI- ---
    plus_di = np.zeros(len(highs))
    minus_di = np.zeros(len(highs))

    for i in range(period, len(highs)): # Valid from 'period' index
        if smoothed_tr[i] != 0:
            plus_di[i] = (smoothed_plus_dm[i] / smoothed_tr[i]) * 100
            minus_di[i] = (smoothed_minus_dm[i] / smoothed_tr[i]) * 100

    # --- Calculate DX ---
    dx_values = np.zeros(len(highs))
    for i in range(period, len(highs)): # Valid from 'period' index
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx_values[i] = (abs(plus_di[i] - minus_di[i]) / di_sum) * 100

    # --- Calculate ADX (smoothed DX) ---
    # The first ADX value is the simple average of DX values over the first 'period'
    # of *valid* DX values. Valid DX values start from index 'period'.
    # So we sum DX values from index 'period' to `2*period - 1`.
    
    adx_values = np.zeros(len(highs))
    
    # Calculate initial ADX (average of first 'period' DX values)
    # These DX values are from index `period` to `2*period - 1`.
    adx_prev = np.sum(dx_values[period : 2 * period]) / period
    adx_values[2 * period - 1] = adx_prev

    # Apply smoothing for subsequent ADX values
    for i in range(2 * period, len(highs)):
        adx_prev = (adx_prev * (period - 1) + dx_values[i]) / period
        adx_values[i] = adx_prev
            
    return adx_values[-1]


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on Bollinger Bands with RSI confirmation
    and a higher-timeframe ADX trend filter.

    A Buy signal is generated if:
    - Primary timeframe price is below the lower Bollinger Band.
    - Primary timeframe RSI indicates an oversold condition.
    - Higher-timeframe ADX indicates a weak or non-trending market.

    A Sell signal is generated if:
    - Primary timeframe price is above the upper Bollinger Band.
    - Primary timeframe RSI indicates an overbought condition.
    - Higher-timeframe ADX indicates a weak or non-trending market.

    Args:
        data (MarketData): A dictionary of market data for various currency pairs.

    Returns:
        list[BuySignal | SellSignal]: A list of generated buy or sell signals.
    """
    signals: list[BuySignal | SellSignal] = []
    
    rule_id = "81eda500-20e9-41a0-a793-ee529e7703bc"

    for pair, pair_data in data.items():
        # Ensure we have enough hot data for current price and timestamp
        # And enough warm candles for all calculations (PTF indicators + HTF ADX)
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES_REQUIRED_TOTAL:
            continue

        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        ts = current_tick.polled_at

        # --- Primary Timeframe (1h) Calculations: Bollinger Bands and RSI ---
        # `warm` data is assumed to be ordered from oldest to newest.
        # We need at least MIN_PRIMARY_TIMEFRAME_CANDLES for BB and RSI.
        ptf_closes = np.array([c.close for c in pair_data.warm[-MIN_PRIMARY_TIMEFRAME_CANDLES:]])

        # Calculate Bollinger Bands
        bb_period_closes = ptf_closes[-BB_PERIOD:]
        mean = np.mean(bb_period_closes)
        std = np.std(bb_period_closes)

        if std == 0: # Price hasn't moved, no meaningful bands
            continue

        lower_band = mean - BB_STD_DEV * std
        upper_band = mean + BB_STD_DEV * std

        # Calculate RSI
        rsi_value = calculate_rsi(ptf_closes, RSI_PERIOD)
        if np.isnan(rsi_value):
            continue

        # --- Higher Timeframe (4h) Calculations: ADX Trend Filter ---
        htf_candles = resample_candles(pair_data.warm, HIGHER_TIMEFRAME_HOURS)

        # Check if enough HTF candles are available for ADX calculation
        if len(htf_candles) < MIN_HTF_CANDLES_FOR_ADX:
            continue
        
        # Extract OHLC for ADX calculation
        htf_highs = np.array([c.high for c in htf_candles])
        htf_lows = np.array([c.low for c in htf_candles])
        htf_closes = np.array([c.close for c in htf_candles])

        htf_adx = calculate_adx(htf_highs, htf_lows, htf_closes, ADX_PERIOD)

        if np.isnan(htf_adx):
            continue
        
        # --- Generate Signals with Combined Conditions ---
        is_weak_trend_htf = htf_adx < ADX_TREND_THRESHOLD

        # Buy Signal Conditions
        if (current_price < lower_band and 
            rsi_value < RSI_OVERSOLD_THRESHOLD and 
            is_weak_trend_htf):
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price, rule_id=rule_id))
        
        # Sell Signal Conditions
        elif (current_price > upper_band and 
              rsi_value > RSI_OVERBOUGHT_THRESHOLD and 
              is_weak_trend_htf):
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price, rule_id=rule_id))

    return signals