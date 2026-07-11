from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Parameters (adjusted for feasibility with 24 hourly candles in `warm` data)
# Original pseudocode: adx_period = 14; ma_period = 50; adx_threshold = 25;
# The `warm` list contains at most 24 hourly candles.
# An `adx_period` of 14 requires ~27 candles for the first ADX value.
# An `ma_period` of 50 requires 50 candles for the first MA value.
# To ensure signals can be generated with the available data, these periods are adjusted.
ADX_PERIOD = 10  # Needs 2*ADX_PERIOD - 1 = 19 candles for the first ADX value.
MA_PERIOD = 20   # Needs MA_PERIOD = 20 candles for the first SMA value.
ADX_THRESHOLD = 25

# Minimum candles required for calculations:
# We need `MA_PERIOD` candles for SMA.
# We need `2 * ADX_PERIOD - 1` candles for ADX.
# Additionally, for the price crossover logic, we need the current and previous candle's MA/close.
# So, we need `max(MA_PERIOD, (2 * ADX_PERIOD - 1)) + 1` candles in total.
MIN_REQUIRED_CANDLES = max(MA_PERIOD, (2 * ADX_PERIOD - 1)) + 1
# For ADX_PERIOD=10, MA_PERIOD=20: max(20, 19) + 1 = 20 + 1 = 21 candles.


def _calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates ADX, +DI, and -DI using Wilder's smoothing method.
    Inputs are numpy arrays of high, low, and close prices.
    Returns ADX, DI_plus, DI_minus as numpy arrays, aligned with the input arrays.
    The first `2*period - 2` values of ADX will be NaN.
    The first `period - 1` values of DI_plus, DI_minus will be NaN.
    """
    if len(high) < period * 2: # Need enough data for initial smoothing periods and ADX calculation
        return np.full_like(close, np.nan), np.full_like(close, np.nan), np.full_like(close, np.nan)

    tr = np.zeros_like(close)
    dm_plus = np.zeros_like(close)
    dm_minus = np.zeros_like(close)

    for i in range(1, len(close)):
        # True Range
        tr_val = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr[i] = tr_val

        # Directional Movement
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]

        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0

        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0

    # Wilder's smoothing for TR, DM_plus, DM_minus
    atr = np.zeros_like(close)
    di_plus_smooth = np.zeros_like(close)
    di_minus_smooth = np.zeros_like(close)

    # First period sum for ATR, DI_plus_smooth, DI_minus_smooth (from index 1 as TR/DM needs prev candle)
    atr[period-1] = np.sum(tr[1:period])
    di_plus_smooth[period-1] = np.sum(dm_plus[1:period])
    di_minus_smooth[period-1] = np.sum(dm_minus[1:period])

    for i in range(period, len(close)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        di_plus_smooth[i] = (di_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
        di_minus_smooth[i] = (di_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period

    # Calculate DI_plus and DI_minus
    di_plus = np.zeros_like(close)
    di_minus = np.zeros_like(close)

    # Avoid division by zero
    valid_atr_indices = atr[period-1:] != 0
    di_plus[period-1:][valid_atr_indices] = (di_plus_smooth[period-1:][valid_atr_indices] / atr[period-1:][valid_atr_indices]) * 100
    di_minus[period-1:][valid_atr_indices] = (di_minus_smooth[period-1:][valid_atr_indices] / atr[period-1:][valid_atr_indices]) * 100

    # Calculate DX
    dx = np.zeros_like(close)
    # Avoid division by zero for DI_plus + DI_minus
    sum_di = di_plus[period-1:] + di_minus[period-1:]
    valid_di_sum_indices = sum_di != 0
    dx[period-1:][valid_di_sum_indices] = np.abs((di_plus[period-1:][valid_di_sum_indices] - di_minus[period-1:][valid_di_sum_indices]) / sum_di[valid_di_sum_indices]) * 100

    # Calculate ADX (smoothed DX)
    adx = np.zeros_like(close)
    
    # First ADX value is the simple average of the first `period` DX values
    # DX values are available from index `period-1`.
    # So, the first ADX value is the mean of DX[period-1] to DX[2*period-2].
    if len(dx) >= 2*period - 1:
        adx[2*period-2] = np.mean(dx[period-1 : 2*period-1])

        for i in range(2*period-1, len(close)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Fill leading NaNs for alignment as ADX, DI_plus, DI_minus are not available for early bars
    adx[:2*period-2] = np.nan
    di_plus[:period-1] = np.nan
    di_minus[:period-1] = np.nan

    return adx, di_plus, di_minus


def _calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates Simple Moving Average (SMA).
    Returns SMA as a numpy array, aligned with the input prices.
    The first `period - 1` values will be NaN.
    """
    if len(prices) < period:
        return np.full_like(prices, np.nan)
    
    sma = np.full_like(prices, np.nan)
    # Calculate initial sum for the first SMA value
    current_sum = np.sum(prices[:period])
    sma[period-1] = current_sum / period

    for i in range(period, len(prices)):
        current_sum = current_sum - prices[i-period] + prices[i]
        sma[i] = current_sum / period
    return sma


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    RULE_ID = "ADX_PriceAction_Trend_Confirmation"

    for pair, pair_data in data.items():
        candles = pair_data.warm
        
        if len(candles) < MIN_REQUIRED_CANDLES:
            continue

        # Ensure candles are sorted by hour (oldest first)
        candles.sort(key=lambda c: c.hour)

        # Extract prices as numpy arrays
        high_prices = np.array([c.high for c in candles])
        low_prices = np.array([c.low for c in candles])
        close_prices = np.array([c.close for c in candles])

        # Calculate Indicators
        adx, di_plus, di_minus = _calculate_adx(high_prices, low_prices, close_prices, ADX_PERIOD)
        ma = _calculate_sma(close_prices, MA_PERIOD)

        # Check if the last candle's indicators are valid (not NaN)
        # We need values for the current and previous candles for the crossover logic
        if np.isnan(adx[-1]) or np.isnan(di_plus[-1]) or np.isnan(di_minus[-1]) or np.isnan(ma[-1]) or \
           np.isnan(ma[-2]) or np.isnan(close_prices[-2]):
            continue # Not enough valid data for the current candle's indicators or previous candle's MA/close

        # Current and Previous values for comparison
        current_close = close_prices[-1]
        previous_close = close_prices[-2] 

        current_adx = adx[-1]
        current_di_plus = di_plus[-1]
        current_di_minus = di_minus[-1]
        current_ma = ma[-1]
        previous_ma = ma[-2]

        # Determine Signals
        if current_adx > ADX_THRESHOLD:
            # Buy signal conditions
            if current_di_plus > current_di_minus:
                # Bullish trend direction
                # Price crosses above MA: current close > current MA AND previous close <= previous MA
                if current_close > current_ma and previous_close <= previous_ma:
                    signals.append(BuySignal(
                        pair=pair,
                        timestamp=candles[-1].hour, # Use candle hour for timestamp
                        price=current_close,
                        rule_id=RULE_ID
                    ))
            # Sell signal conditions
            elif current_di_minus > current_di_plus:
                # Bearish trend direction
                # Price crosses below MA: current close < current MA AND previous close >= previous MA
                if current_close < current_ma and previous_close >= previous_ma:
                    signals.append(SellSignal(
                        pair=pair,
                        timestamp=candles[-1].hour, # Use candle hour for timestamp
                        price=current_close,
                        rule_id=RULE_ID
                    ))
    return signals