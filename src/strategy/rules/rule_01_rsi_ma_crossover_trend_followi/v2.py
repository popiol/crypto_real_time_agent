from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

# Rule specific constants
RULE_ID = "RSI_SMA_Relax_001"
RSI_PERIOD = 14
SMA_SHORT_PERIOD = 20  # Modified from 10 to 20
SMA_LONG_PERIOD = 50   # Modified from 30 to 50
RSI_OVERSOLD = 40      # Modified from 30 to 40
RSI_OVERBOUGHT = 60    # Modified from 70 to 60

# Minimum data required for calculations.
# RSI(14) needs (period + 2) candles for current and previous RSI for crossover detection. (14+2=16)
# SMA(20) needs (period + 1) candles for current and previous SMA for crossover detection. (20+1=21)
# SMA(50) needs (period + 1) candles for current and previous SMA for crossover detection. (50+1=51)
# The maximum of these is 51.
MIN_CANDLES_REQUIRED = max(RSI_PERIOD + 2, SMA_SHORT_PERIOD + 1, SMA_LONG_PERIOD + 1)


def _calculate_rsi(prices: np.ndarray, period: int) -> tuple[float | None, float | None]:
    """
    Calculates the current and previous RSI values for the given prices.
    Returns (current_rsi, previous_rsi).
    Requires at least `period + 2` prices to calculate both current and previous RSI values.
    """
    if len(prices) < period + 2:
        return None, None

    # Calculate price changes (deltas)
    # deltas[i] = prices[i+1] - prices[i]
    deltas = np.diff(prices)

    # Gains and Losses based on deltas
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsis = np.full(len(prices), np.nan)

    # Initial average gain/loss for the first `period` deltas
    # (i.e., corresponding to prices[0]...prices[period])
    if len(gains) < period:  # Not enough deltas for the first average
        return None, None

    avg_gain_prev = np.mean(gains[:period])
    avg_loss_prev = np.mean(losses[:period])

    # Calculate first RSI value (for prices[period])
    if avg_loss_prev == 0:
        rs = np.inf
    else:
        rs = avg_gain_prev / avg_loss_prev
    rsis[period] = 100 - (100 / (1 + rs)) if not np.isinf(rs) else 100

    # Exponential smoothing for subsequent periods
    # Loop starts from the price index `period + 1` up to the last price.
    for i in range(period + 1, len(prices)):
        # current_gain/loss corresponds to the delta between prices[i] and prices[i-1]
        current_gain = gains[i - 1]
        current_loss = losses[i - 1]

        avg_gain_curr = ((avg_gain_prev * (period - 1)) + current_gain) / period
        avg_loss_curr = ((avg_loss_prev * (period - 1)) + current_loss) / period

        if avg_loss_curr == 0:
            rs = np.inf
        else:
            rs = avg_gain_curr / avg_loss_curr

        rsis[i] = 100 - (100 / (1 + rs)) if not np.isinf(rs) else 100

        # Update for next iteration
        avg_gain_prev = avg_gain_curr
        avg_loss_prev = avg_loss_curr

    # Return current_rsi (last calculated) and previous_rsi (second to last calculated)
    # Ensure both values are valid (not NaN)
    if not np.isnan(rsis[-1]) and not np.isnan(rsis[-2]):
        return float(rsis[-1]), float(rsis[-2])
    return None, None


def _calculate_sma(prices: np.ndarray, period: int) -> tuple[float | None, float | None]:
    """
    Calculates the current and previous SMA values for the given prices.
    Returns (current_sma, previous_sma).
    Requires at least `period + 1` prices to calculate both current and previous SMA values.
    """
    if len(prices) < period + 1:
        return None, None

    # Current SMA is the mean of the last `period` prices
    current_sma = np.mean(prices[-period:])
    # Previous SMA is the mean of the `period` prices ending one step before the latest
    previous_sma = np.mean(prices[-(period + 1):-1])

    return float(current_sma), float(previous_sma)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Constraint: data.warm holds at most 24 hourly candles.
        # This rule requires MIN_CANDLES_REQUIRED = 51 candles.
        # Therefore, this rule will never generate signals under the given data constraints
        # if only `data.warm` is used, as 24 < 51.
        # It handles "insufficient data gracefully" by returning an empty list for any pair.
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        # Extract close prices for calculations
        close_prices = np.array([c.close for c in warm_candles])

        # Get the latest timestamp and price for signal generation
        latest_candle = warm_candles[-1]
        latest_timestamp = latest_candle.hour
        latest_price = latest_candle.close

        # Calculate indicators
        current_rsi, prev_rsi = _calculate_rsi(close_prices, RSI_PERIOD)
        current_sma_short, prev_sma_short = _calculate_sma(close_prices, SMA_SHORT_PERIOD)
        current_sma_long, prev_sma_long = _calculate_sma(close_prices, SMA_LONG_PERIOD)

        # If any indicator cannot be calculated, skip this pair
        if any(val is None for val in [current_rsi, prev_rsi, current_sma_short, prev_sma_short, current_sma_long, prev_sma_long]):
            continue

        # Determine crossover conditions for RSI and SMAs
        # RSI crosses above RSI_OVERSOLD (40): (previous <= 40) AND (current > 40)
        rsi_cross_above_oversold = (prev_rsi <= RSI_OVERSOLD) and (current_rsi > RSI_OVERSOLD)
        # RSI crosses below RSI_OVERBOUGHT (60): (previous >= 60) AND (current < 60)
        rsi_cross_below_overbought = (prev_rsi >= RSI_OVERBOUGHT) and (current_rsi < RSI_OVERBOUGHT)

        # Short SMA crosses above Long SMA: (previous_short <= previous_long) AND (current_short > current_long)
        sma_cross_above = (prev_sma_short <= prev_sma_long) and (current_sma_short > current_sma_long)
        # Short SMA crosses below Long SMA: (previous_short >= previous_long) AND (current_short < current_long)
        sma_cross_below = (prev_sma_short >= prev_sma_long) and (current_sma_short < current_sma_long)

        # --- Entry Signals ---
        # IF RSI(14) crosses above 40 AND SMA(20) crosses above SMA(50) THEN BUY
        if rsi_cross_above_oversold and sma_cross_above:
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_timestamp,
                price=latest_price,
                rule_id=RULE_ID,
                confidence=1.0  # High confidence for entry
            ))
        # ELSE IF RSI(14) crosses below 60 AND SMA(20) crosses below SMA(50) THEN SELL
        elif rsi_cross_below_overbought and sma_cross_below:
            signals.append(SellSignal(
                pair=pair,
                timestamp=latest_timestamp,
                price=latest_price,
                rule_id=RULE_ID,
                confidence=1.0  # High confidence for entry
            ))

        # --- Exit Signals (Adapted from original rule, with new thresholds) ---
        # IF (CURRENTLY_LONG AND (RSI(14) crosses above 60 OR SMA(20) crosses below SMA(50))) THEN CLOSE_LONG_POSITION()
        # The `signal` function is stateless, so it cannot know `CURRENTLY_LONG`.
        # It generates a SellSignal if the conditions for closing a long position are met.
        rsi_cross_above_overbought_for_exit = (prev_rsi <= RSI_OVERBOUGHT) and (current_rsi > RSI_OVERBOUGHT)
        if rsi_cross_above_overbought_for_exit or sma_cross_below:
            signals.append(SellSignal(
                pair=pair,
                timestamp=latest_timestamp,
                price=latest_price,
                rule_id=RULE_ID,
                confidence=0.8  # Lower confidence for exit
            ))

        # IF (CURRENTLY_SHORT AND (RSI(14) crosses below 40 OR SMA(20) crosses above SMA(50))) THEN CLOSE_SHORT_POSITION()
        # The `signal` function is stateless, so it cannot know `CURRENTLY_SHORT`.
        # It generates a BuySignal if the conditions for closing a short position are met.
        rsi_cross_below_oversold_for_exit = (prev_rsi >= RSI_OVERSOLD) and (current_rsi < RSI_OVERSOLD)
        if rsi_cross_below_oversold_for_exit or sma_cross_above:
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_timestamp,
                price=latest_price,
                rule_id=RULE_ID,
                confidence=0.8  # Lower confidence for exit
            ))

    return signals