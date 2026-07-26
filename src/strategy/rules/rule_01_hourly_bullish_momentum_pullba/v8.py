from __future__ import annotations
from datetime import datetime
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

RULE_ID = "relaxed_stoch_pullback_v2"

# Constants
STOCH_PERIOD = 14
STOCH_SMOOTH_K = 3  # This is the smoothing period for %K itself, effectively creating a "Slow Stochastic %K".
CANDLE_BODY_RATIO_THRESHOLD = 0.5
STOCH_K_NOT_OVERBOUGHT_THRESHOLD = 70
STOCH_K_OVERBOUGHT_SELL_THRESHOLD = 80
STOP_LOSS_PERCENT = 0.02
PROFIT_TARGET_PERCENT = 0.03

def _calculate_stoch_k_series(
    warm_candles: list[WarmCandle], period: int, smooth_k: int
) -> list[float]:
    """
    Calculates the smoothed %K values for a series of warm candles.
    %K is calculated as a `smooth_k`-period Simple Moving Average (SMA) of the raw
    `period`-period %K.
    Returns a list of calculated %K values, or an empty list if insufficient data
    to calculate any valid smoothed %K.
    """
    # Minimum candles needed to calculate the first raw %K
    if len(warm_candles) < period:
        return []

    closes = [c.close for c in warm_candles]
    highs = [c.high for c in warm_candles]
    lows = [c.low for c in warm_candles]

    raw_k_values = []
    # Calculate raw %K values for all possible candles
    for i in range(period - 1, len(warm_candles)):  # Start from the first index where raw K is possible
        window_start_idx = i - period + 1
        current_window_lows = lows[window_start_idx : i + 1]
        current_window_highs = highs[window_start_idx : i + 1]
        current_close = closes[i]

        lowest_low = min(current_window_lows)
        highest_high = max(current_window_highs)

        # Avoid division by zero if range is zero, return a neutral value
        if (highest_high - lowest_low) == 0:
            raw_k_values.append(50.0)
        else:
            raw_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            raw_k_values.append(raw_k)
            
    # Now smooth the raw %K values
    smoothed_k_values = []
    # Minimum raw K values needed to calculate the first smoothed %K
    if len(raw_k_values) < smooth_k:
        return []

    for i in range(smooth_k - 1, len(raw_k_values)):  # Start from the first index where smoothed K is possible
        k_slice = raw_k_values[i - smooth_k + 1 : i + 1]
        smoothed_k_values.append(np.mean(k_slice))
            
    return smoothed_k_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Access current_position from data.positions; it will be None if no open position
        current_position = data.positions.get(pair)

        # --- Initial Data Checks ---
        # Need at least one hot tick for current price and timestamp
        if not pair_data.hot:
            continue
        
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_timestamp = current_tick.polled_at

        # For buy signal, we need `last_candle` (data.warm[-1]), so at least 1 candle.
        # Stochastic calculation requires a minimum number of candles; _calculate_stoch_k_series handles this.
        if not pair_data.warm:
            continue

        last_candle = pair_data.warm[-1]

        # --- Calculate Stochastic %K series ---
        stoch_k_series = _calculate_stoch_k_series(
            pair_data.warm, STOCH_PERIOD, STOCH_SMOOTH_K
        )
        
        # If stoch_k_series is empty, it means there wasn't enough data for a valid smoothed %K
        if not stoch_k_series:
            continue

        stoch_k_current = stoch_k_series[-1]
        
        # --- Buy Signal Logic ---
        # 1. Bullish Candle: last_candle.close > last_candle.open
        is_bullish_candle = last_candle.close > last_candle.open_price

        # 2. Decent Body: abs(last_candle.close - last_candle.open) / candle_range > CANDLE_BODY_RATIO_THRESHOLD
        candle_range = last_candle.high - last_candle.low
        candle_body_ratio = 0.0
        if candle_range > 0:  # Avoid division by zero
            candle_body_ratio = abs(last_candle.close - last_candle.open_price) / candle_range
        has_decent_body = candle_body_ratio > CANDLE_BODY_RATIO_THRESHOLD

        # 3. Pullback: current_price < last_candle.high
        has_pullback = current_price < last_candle.high

        # 4. Stochastic Not Overbought: stoch_k_current < STOCH_K_NOT_OVERBOUGHT_THRESHOLD
        is_stoch_not_overbought = stoch_k_current < STOCH_K_NOT_OVERBOUGHT_THRESHOLD

        if (
            is_bullish_candle
            and has_decent_body
            and has_pullback
            and is_stoch_not_overbought
        ):
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=RULE_ID,
                confidence=1.0,
            ))

        # --- Sell Signal Logic (if there's an open position) ---
        if current_position:
            entry_price = current_position.entry_price

            # 1. Stop Loss
            if current_price < entry_price * (1 - STOP_LOSS_PERCENT):
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=1.0,
                ))
            # 2. Profit Target
            elif current_price > entry_price * (1 + PROFIT_TARGET_PERCENT):
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=1.0,
                ))
            # 3. Stochastic Turn Down from Overbought
            # Need at least 2 smoothed K values to detect a turn down.
            # _calculate_stoch_k_series returns only valid calculated values, so len() >= 2 is sufficient.
            elif len(stoch_k_series) >= 2:
                stoch_k_prev = stoch_k_series[-2]
                if (
                    stoch_k_current > STOCH_K_OVERBOUGHT_SELL_THRESHOLD
                    and stoch_k_current < stoch_k_prev
                ):
                    signals.append(SellSignal(
                        pair=pair,
                        timestamp=current_timestamp,
                        price=current_price,
                        rule_id=RULE_ID,
                        confidence=1.0,
                    ))

    return signals