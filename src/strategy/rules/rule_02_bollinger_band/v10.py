from __future__ import annotations

import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle, ColdMonth


# Rule parameters
PERIOD_BB = 20  # Period for Bollinger Bands SMA and Standard Deviation
STD_DEV_BB = 2.0  # Multiplier for Bollinger Band standard deviation
PERIOD_RSI = 14  # Period for Relative Strength Index (RSI) calculation
LOOKBACK_RSI_ADAPTIVE = 60  # Number of past RSI values to use for adaptive thresholds
K_RSI_STD_DEV = 1.0  # Multiplier for RSI standard deviation in adaptive thresholds

# Minimum candles required for calculations:
# 1. Bollinger Bands: needs PERIOD_BB candles.
# 2. RSI series: to get N RSI values, we need N + PERIOD_RSI candles.
#    We need LOOKBACK_RSI_ADAPTIVE RSI values to calculate adaptive thresholds.
#    So, we need LOOKBACK_RSI_ADAPTIVE + PERIOD_RSI candles in total for RSI history.
MIN_CANDLES_REQUIRED = max(PERIOD_BB, PERIOD_RSI + LOOKBACK_RSI_ADAPTIVE)


def calculate_rsi_series(prices: list[float], period: int) -> list[float]:
    """
    Calculates the Relative Strength Index (RSI) for a list of prices.
    Returns a list of RSI values, starting from the (period)-th price point.
    """
    if len(prices) < period + 1:
        return []

    rsi_values = []
    gains = [0.0] * len(prices)
    losses = [0.0] * len(prices)

    # Calculate initial gains/losses
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = abs(change)

    # Calculate initial average gain/loss for the first 'period' changes
    # (from index 1 to period, corresponding to prices[0] to prices[period])
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period

    if avg_loss == 0:
        # If no losses in the initial period, RSI is 100.
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))

    # Calculate subsequent RSI values using Wilder's smoothing method
    for i in range(period + 1, len(prices)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

        if avg_loss == 0:
            # If current avg_loss is zero, RSI is 100
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

    return rsi_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm candle data for all calculations
        # Need at least MIN_CANDLES_REQUIRED for BB, RSI, and adaptive thresholds
        if not pair_data.warm or len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue

        closes = [c.close for c in pair_data.warm]

        # 1. Calculate Bollinger Bands
        # Use the last PERIOD_BB closes for BB calculation
        bb_closes = closes[-PERIOD_BB:]
        # This check is technically redundant if MIN_CANDLES_REQUIRED is set correctly,
        # but provides an extra safeguard.
        if len(bb_closes) < PERIOD_BB:
            continue

        mean_bb = statistics.mean(bb_closes)

        # Avoid statistics.stdev for single-element lists or all identical values
        if len(bb_closes) > 1:
            std_dev_bb = statistics.stdev(bb_closes)
        else:
            std_dev_bb = 0.0

        if std_dev_bb == 0:
            # If standard deviation is zero, all prices are the same, BBs collapse.
            # This makes signals impossible or meaningless for a band strategy. Skip.
            continue

        upper_bb = mean_bb + (STD_DEV_BB * std_dev_bb)
        lower_bb = mean_bb - (STD_DEV_BB * std_dev_bb)

        # 2. Calculate RSI series
        # The `closes` list should be long enough based on MIN_CANDLES_REQUIRED
        rsi_series = calculate_rsi_series(closes, PERIOD_RSI)

        # We need at least LOOKBACK_RSI_ADAPTIVE RSI values to calculate adaptive thresholds
        if len(rsi_series) < LOOKBACK_RSI_ADAPTIVE:
            continue

        current_rsi = rsi_series[-1]

        # 3. Calculate Adaptive RSI Thresholds
        # Use the last LOOKBACK_RSI_ADAPTIVE RSI values to determine thresholds
        rsi_history_for_adaptive = rsi_series[-LOOKBACK_RSI_ADAPTIVE:]

        rsi_mean = statistics.mean(rsi_history_for_adaptive)

        # Handle case where all RSI values are identical in the lookback window
        # (e.g., if market is completely flat for a long time, RSI could be 50 constantly)
        if len(rsi_history_for_adaptive) > 1:  # stdev requires at least 2 points
            rsi_std_dev = statistics.stdev(rsi_history_for_adaptive)
        else:  # If only one RSI value, std dev is 0
            rsi_std_dev = 0.0

        adaptive_oversold_threshold = rsi_mean - (K_RSI_STD_DEV * rsi_std_dev)
        adaptive_overbought_threshold = rsi_mean + (K_RSI_STD_DEV * rsi_std_dev)

        # Ensure thresholds are within valid RSI range (0-100)
        adaptive_oversold_threshold = max(0.0, min(100.0, adaptive_oversold_threshold))
        adaptive_overbought_threshold = max(0.0, min(100.0, adaptive_overbought_threshold))

        # Adjust if overbought threshold becomes less than oversold threshold (e.g., due to clamping or zero std_dev)
        if adaptive_overbought_threshold < adaptive_oversold_threshold:
            # If they cross, set them to a small range around the mean to maintain logic.
            mid_point = (adaptive_oversold_threshold + adaptive_overbought_threshold) / 2
            # Use a small constant delta, ensuring it doesn't push them out of 0-100 range
            delta = 0.5  # A small fixed delta
            adaptive_oversold_threshold = max(0.0, mid_point - delta)
            adaptive_overbought_threshold = min(100.0, mid_point + delta)
            # Re-check if they still crossed due to clamping, then set to identical mean
            if adaptive_overbought_threshold < adaptive_oversold_threshold:
                adaptive_oversold_threshold = mid_point
                adaptive_overbought_threshold = mid_point

        # 4. Generate Signals
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # Buy Signal Condition: Price below lower BB AND RSI below adaptive oversold threshold
        if current_price < lower_bb and current_rsi < adaptive_oversold_threshold:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Sell Signal Condition: Price above upper BB AND RSI above adaptive overbought threshold
        elif current_price > upper_bb and current_rsi > adaptive_overbought_threshold:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals