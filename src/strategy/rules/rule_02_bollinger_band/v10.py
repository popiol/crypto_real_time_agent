from __future__ import annotations

import numpy as np
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# --- Parameters for Bollinger Bands ---
BB_PERIOD = 20  # Number of candles for Bollinger Band calculation (e.g., 20 for 20-period SMA)
BB_STD_DEV = 2.0  # Standard deviation multiplier for Bollinger Bands (e.g., 2 standard deviations)

# --- Parameters for RSI ---
RSI_PERIOD = 14  # Number of candles for RSI calculation (e.g., 14 periods)
RSI_OVERSOLD_THRESHOLD = 30  # RSI level below which an asset is considered oversold
RSI_OVERBOUGHT_THRESHOLD = 70  # RSI level above which an asset is considered overbought

# --- Parameters for ADX ---
ADX_PERIOD = 14  # Number of periods for ADX calculation
ADX_NON_TREND_THRESHOLD = 25  # ADX level below which market is considered non-trending/ranging

# Minimum candles required for calculations:
# BB_PERIOD candles for BB's SMA and StdDev.
# RSI_PERIOD + 1 candles for RSI (to get RSI_PERIOD price changes).
# ADX needs 2 * ADX_PERIOD candles for the first ADX value (period for initial TR/DM sums, then period for DX sums).
MIN_CANDLES_REQUIRED = max(BB_PERIOD, RSI_PERIOD + 1, 2 * ADX_PERIOD)


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

    # Calculate initial average gain and loss over the first 'period' differences
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Handle cases where initial avg_loss is zero
    if avg_loss == 0:
        rs = np.inf if avg_gain > 0 else 1.0  # If avg_gain > 0, RSI is 100. If both 0, RSI is 50.
    else:
        rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # Apply smoothing for subsequent periods if more data points are available
    for i in range(period, len(gains)):
        current_gain = gains[i]
        current_loss = losses[i]

        # RSI smoothing formula (Wilder's smoothing)
        avg_gain = ((avg_gain * (period - 1)) + current_gain) / period
        avg_loss = ((avg_loss * (period - 1)) + current_loss) / period

        if avg_loss == 0:
            rs = np.inf if avg_gain > 0 else 1.0
        else:
            rs = avg_gain / avg_loss
        
        rsi = 100 - (100 / (1 + rs))
            
    return rsi


def calculate_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
    """
    Calculates the Average Directional Index (ADX).
    Assumes arrays are ordered from oldest to newest.

    Args:
        highs (np.ndarray): Array of high prices.
        lows (np.ndarray): Array of low prices.
        closes (np.ndarray): Array of closing prices.
        period (int): The number of periods for ADX calculation.

    Returns:
        float: The latest ADX value, or np.nan if not enough data.
    """
    n = len(closes)
    # ADX requires at least 2 * period candles for the first valid ADX value
    if n < 2 * period:
        return np.nan

    # Arrays for True Range (TR), Positive Directional Movement (DM+), Negative Directional Movement (DM-)
    tr = np.zeros(n)
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)

    for i in range(1, n):
        # True Range
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))

        # Directional Movement
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]

        dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0

    # Arrays for Smoothed TR, DM+, DM-
    smooth_tr = np.zeros(n)
    smooth_dm_plus = np.zeros(n)
    smooth_dm_minus = np.zeros(n)

    # Initial sum for the first 'period' values (from index 1 to period)
    # The first smoothed value will be stored at index 'period'
    smooth_tr[period] = np.sum(tr[1 : period + 1])
    smooth_dm_plus[period] = np.sum(dm_plus[1 : period + 1])
    smooth_dm_minus[period] = np.sum(dm_minus[1 : period + 1])

    # Apply Wilder's smoothing from period + 1 onwards
    for i in range(period + 1, n):
        smooth_tr[i] = smooth_tr[i-1] - (smooth_tr[i-1] / period) + tr[i]
        smooth_dm_plus[i] = smooth_dm_plus[i-1] - (smooth_dm_plus[i-1] / period) + dm_plus[i]
        smooth_dm_minus[i] = smooth_dm_minus[i-1] - (smooth_dm_minus[i-1] / period) + dm_minus[i]

    # Directional Indicators (DI)
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)

    # DI values are valid from index 'period'
    for i in range(period, n):
        if smooth_tr[i] == 0:
            di_plus[i] = 0
            di_minus[i] = 0
        else:
            di_plus[i] = (smooth_dm_plus[i] / smooth_tr[i]) * 100
            di_minus[i] = (smooth_dm_minus[i] / smooth_tr[i]) * 100

    # Directional Movement Index (DX)
    dx = np.zeros(n)
    # DX values are valid from index 'period'
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum == 0:
            dx[i] = 0
        else:
            dx[i] = (abs(di_plus[i] - di_minus[i]) / di_sum) * 100

    # Average Directional Index (ADX)
    adx_values = np.zeros(n)

    # The first ADX value is the average of the first 'period' DX values
    # These DX values are from index 'period' to '2*period - 1'
    adx_values[2 * period - 1] = np.mean(dx[period : 2 * period])

    # Apply Wilder's smoothing for ADX from 2*period onwards
    for i in range(2 * period, n):
        adx_values[i] = (adx_values[i-1] * (period - 1) + dx[i]) / period

    return adx_values[n-1]


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on Bollinger Bands with RSI confirmation,
    filtered by ADX to only trade in non-trending markets.

    A Buy signal is generated when the price drops below the lower Bollinger Band,
    the RSI indicates an oversold condition (below RSI_OVERSOLD_THRESHOLD),
    and ADX indicates a non-trending market (below ADX_NON_TREND_THRESHOLD).

    A Sell signal is generated when the price rises above the upper Bollinger Band,
    the RSI indicates an overbought condition (above RSI_OVERBOUGHT_THRESHOLD),
    and ADX indicates a non-trending market (below ADX_NON_TREND_THRESHOLD).

    Args:
        data (MarketData): A dictionary of market data for various currency pairs.

    Returns:
        list[BuySignal | SellSignal]: A list of generated buy or sell signals.
    """
    signals: list[BuySignal | SellSignal] = []
    
    # Define rule_id for generated signals based on the provided idea_id
    rule_id = "b75bf204-08a9-488d-a498-9c6d604ce9d5"

    for pair, pair_data in data.items():
        # Ensure we have enough hot data for current price and timestamp
        # And enough warm candles for BB, RSI, and ADX calculations
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue

        # Get the latest 'hot' tick for the current price and timestamp
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        ts = current_tick.polled_at

        # --- Prepare data for Bollinger Bands, RSI, and ADX ---
        # Take the most recent 'MIN_CANDLES_REQUIRED' candles from warm data.
        # 'warm' data is assumed to be ordered from oldest to newest.
        relevant_candles = pair_data.warm[-MIN_CANDLES_REQUIRED:]
        
        closes = np.array([c.close for c in relevant_candles])
        highs = np.array([c.high for c in relevant_candles])
        lows = np.array([c.low for c in relevant_candles])

        # --- Calculate Bollinger Bands ---
        bb_period_closes = closes[-BB_PERIOD:]
        mean = np.mean(bb_period_closes)
        std = np.std(bb_period_closes)

        # If standard deviation is zero, the price hasn't moved, so no band (or flat band).
        # This indicates a non-tradable condition, so we skip.
        if std == 0:
            continue

        lower_band = mean - BB_STD_DEV * std
        upper_band = mean + BB_STD_DEV * std

        # --- Calculate RSI ---
        rsi_closes = closes[-(RSI_PERIOD + 1):]
        rsi_value = calculate_rsi(rsi_closes, RSI_PERIOD)

        # If RSI could not be calculated, skip.
        if np.isnan(rsi_value):
            continue

        # --- Calculate ADX ---
        adx_value = calculate_adx(highs, lows, closes, ADX_PERIOD)

        # If ADX could not be calculated (e.g., due to insufficient data), skip.
        if np.isnan(adx_value):
            continue

        # --- Check market regime ---
        is_non_trending = (adx_value < ADX_NON_TREND_THRESHOLD)

        # --- Generate Signals with RSI and ADX Confirmation ---
        if is_non_trending:
            if current_price < lower_band and rsi_value < RSI_OVERSOLD_THRESHOLD:
                signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price, rule_id=rule_id))
            elif current_price > upper_band and rsi_value > RSI_OVERBOUGHT_THRESHOLD:
                signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price, rule_id=rule_id))

    return signals