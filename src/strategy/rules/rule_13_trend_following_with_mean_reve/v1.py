from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Parameters for the Trend-following with Mean-Reversion Pullback rule
LONG_MA_PERIOD = 200  # Period for the long-term Moving Average (e.g., 200-period SMA)
SHORT_MA_PERIOD = 50  # Period for the short-term Moving Average (e.g., 50-period SMA)
TREND_STRENGTH_THRESHOLD = 0.005  # Price must be 0.5% above/below long_ma for trend
PULLBACK_TOLERANCE = 0.001       # Price must touch within 0.1% of short_ma
LOOKBACK_PERIOD = 10             # Periods to check for pullback touch
RULE_ID = "ea5fda52-6148-465a-befc-1b2e4722901b" # Unique identifier for this rule

def _calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculates Simple Moving Average (SMA) for a given period.
    Returns an array of the same length as prices, with NaN for initial periods
    where an SMA cannot be calculated.
    """
    if len(prices) < period:
        # If not enough data, return an array of NaNs matching the input length
        return np.full(len(prices), np.nan)

    weights = np.ones(period) / period
    # Convolve with 'valid' mode to get SMA values where the window fully overlaps
    sma_valid = np.convolve(prices, weights, mode='valid')

    # Pad the beginning with NaNs so the SMA values align with the end of their respective periods
    padded_sma = np.full(len(prices), np.nan)
    padded_sma[period - 1:] = sma_valid
    return padded_sma

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Trend-following with Mean-Reversion Pullback' trading rule.

    This rule identifies a strong prevailing trend using a long-period Moving Average (MA).
    It generates a Buy signal when the price is significantly above the long-period MA
    (uptrend) but pulls back to touch or briefly cross below a shorter-period MA,
    indicating a temporary dip within the uptrend. Conversely, it generates a Sell
    signal when the price is significantly below the long-period MA (downtrend)
    but pulls back to touch or briefly cross above a shorter-period MA, indicating
    a temporary rally within the downtrend.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        if not warm_candles:
            continue

        # Extract close prices and timestamps from WarmCandle data
        close_prices = np.array([c.close for c in warm_candles], dtype=np.float64)
        timestamps = [c.hour for c in warm_candles]

        # Determine the minimum number of candles required for all calculations.
        # We need enough candles for the longest MA (LONG_MA_PERIOD)
        # and for the shortest MA plus the lookback window for historical checks.
        # The earliest historical short_ma needed is for `LOOKBACK_PERIOD` candles ago,
        # which itself requires `SHORT_MA_PERIOD` candles to be calculated.
        required_history = max(LONG_MA_PERIOD, SHORT_MA_PERIOD + LOOKBACK_PERIOD)
        if len(close_prices) < required_history:
            continue # Not enough historical data to apply the rule

        # Calculate the long and short Moving Averages for the entire series
        long_ma_series = _calculate_sma(close_prices, LONG_MA_PERIOD)
        short_ma_series = _calculate_sma(close_prices, SHORT_MA_PERIOD)

        # Ensure the latest MA values are valid (not NaN).
        # This check should pass if `required_history` is sufficient.
        if np.isnan(long_ma_series[-1]) or np.isnan(short_ma_series[-1]):
            continue

        # Get current price and MA values
        current_price = close_prices[-1]
        latest_timestamp = timestamps[-1]
        current_long_ma = long_ma_series[-1]
        current_short_ma = short_ma_series[-1]

        # Determine the prevailing trend based on the long MA
        is_uptrend = current_price > current_long_ma * (1 + TREND_STRENGTH_THRESHOLD)
        is_downtrend = current_price < current_long_ma * (1 - TREND_STRENGTH_THRESHOLD)

        # Check for Buy signal conditions
        if is_uptrend:
            has_pulled_back_buy = False
            # Look back for a pullback to the short MA within the LOOKBACK_PERIOD
            for i in range(1, LOOKBACK_PERIOD + 1):
                history_idx = len(close_prices) - 1 - i # Index for historical candle
                
                # Ensure index is valid and historical short MA is calculated
                if history_idx < 0 or np.isnan(short_ma_series[history_idx]):
                    continue

                historical_price = close_prices[history_idx]
                historical_short_ma = short_ma_series[history_idx]

                # Check if the historical price touched or briefly crossed the historical short MA
                # (i.e., it was within the defined tolerance band)
                if (historical_price <= historical_short_ma * (1 + PULLBACK_TOLERANCE) and
                    historical_price >= historical_short_ma * (1 - PULLBACK_TOLERANCE)):
                    has_pulled_back_buy = True
                    break # Found a qualifying pullback, no need to check further back

            # If a pullback occurred and the current price is now above the short MA, generate Buy signal
            if has_pulled_back_buy and current_price > current_short_ma:
                signals.append(BuySignal(
                    pair=pair,
                    timestamp=latest_timestamp,
                    price=current_price,
                    rule_id=RULE_ID
                ))

        # Check for Sell signal conditions
        if is_downtrend: # Note: 'if' not 'elif' as per pseudocode structure
            has_pulled_back_sell = False
            # Look back for a pullback to the short MA within the LOOKBACK_PERIOD
            for i in range(1, LOOKBACK_PERIOD + 1):
                history_idx = len(close_prices) - 1 - i # Index for historical candle

                # Ensure index is valid and historical short MA is calculated
                if history_idx < 0 or np.isnan(short_ma_series[history_idx]):
                    continue

                historical_price = close_prices[history_idx]
                historical_short_ma = short_ma_series[history_idx]

                # Check if the historical price touched or briefly crossed the historical short MA
                if (historical_price >= historical_short_ma * (1 - PULLBACK_TOLERANCE) and
                    historical_price <= historical_short_ma * (1 + PULLBACK_TOLERANCE)):
                    has_pulled_back_sell = True
                    break # Found a qualifying pullback

            # If a pullback occurred and the current price is now below the short MA, generate Sell signal
            if has_pulled_back_sell and current_price < current_short_ma:
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=latest_timestamp,
                    price=current_price,
                    rule_id=RULE_ID
                ))

    return signals