from __future__ import annotations

import statistics
import numpy as np
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


# Constants for Bollinger Bands
K = 2.0
BB_MIN_CANDLES = 10  # Minimum candles required for Bollinger Band calculation

# Constants for Higher Timeframe (HTF) Trend Confirmation
HTF_EMA_PERIOD = 20  # Period for Exponential Moving Average on the higher timeframe (hourly candles)
HTF_MIN_CANDLES = HTF_EMA_PERIOD  # Minimum candles required for HTF EMA calculation

# Rule ID as provided in the idea
RULE_ID = "a3ae1446-6d4c-4ac9-8b65-61b07057c256"


def calculate_ema(prices: list[float], period: int) -> float | None:
    """
    Calculates the Exponential Moving Average (EMA) for a list of prices.
    Returns the latest EMA value, or None if not enough data.
    """
    if len(prices) < period:
        return None

    # Calculate the Simple Moving Average (SMA) for the first 'period' values
    # to serve as the initial EMA.
    ema = sum(prices[:period]) / period

    alpha = 2 / (period + 1)

    # Apply the EMA formula for subsequent values
    for i in range(period, len(prices)):
        ema = (prices[i] - ema) * alpha + ema

    return ema


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough hot data (at least one tick) and warm data
        # for both Bollinger Bands and HTF EMA calculations.
        min_required_warm_candles = max(BB_MIN_CANDLES, HTF_MIN_CANDLES)
        if not pair_data.hot or len(pair_data.warm) < min_required_warm_candles:
            continue

        warm_closes = [c.close for c in pair_data.warm]

        # --- Bollinger Band Calculation ---
        # (Using the full available warm_closes list which is guaranteed to be >= BB_MIN_CANDLES)
        bb_mean = statistics.mean(warm_closes)
        bb_std = statistics.stdev(warm_closes)

        if bb_std == 0:
            # If standard deviation is zero, prices haven't moved, bands are meaningless.
            continue

        lower_band = bb_mean - K * bb_std
        upper_band = bb_mean + K * bb_std

        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # --- Higher Timeframe Trend Confirmation ---
        # (Using the full available warm_closes list which is guaranteed to be >= HTF_MIN_CANDLES)
        higher_timeframe_ema = calculate_ema(warm_closes, HTF_EMA_PERIOD)

        if higher_timeframe_ema is None:
            # This case should ideally be caught by the initial min_required_warm_candles check,
            # but serves as a safeguard.
            continue

        # The 'last_higher_timeframe_close_price' is the latest hourly close price
        # available in our warm data.
        last_higher_timeframe_close_price = warm_closes[-1]

        # --- Combined Signal Logic ---
        # Buy signal: Price drops below lower BB AND HTF trend is up (close > EMA)
        buy_condition = (current_price < lower_band) and \
                        (last_higher_timeframe_close_price > higher_timeframe_ema)
        
        # Sell signal: Price rises above upper BB AND HTF trend is down (close < EMA)
        sell_condition = (current_price > upper_band) and \
                         (last_higher_timeframe_close_price < higher_timeframe_ema)

        if buy_condition:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price, rule_id=RULE_ID))
        elif sell_condition:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price, rule_id=RULE_ID))

    return signals