from __future__ import annotations
from datetime import datetime
import statistics
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

RULE_ID = "SMA_9_Filtered_Momentum"
SMA_PERIOD = 9
RSI_PERIOD = 14
BB_PERIOD = 14
BB_K = 2.0  # Standard multiplier for Bollinger Bands

# Minimum candles for warm data to calculate all indicators:
# - SMA(9) for `current_sma_9` needs 9 candles.
# - SMA(9) for `previous_sma_9` needs `hourly_closes[-10:-1]`, which means `len(hourly_closes)` must be at least 10.
# - RSI(14) needs at least 2 candles to calculate price changes; our implementation adjusts the initial average period if fewer than 14 changes are available, but requires at least 2.
# - Bollinger Bands (14-period) needs `len(hourly_closes) >= 14`.
# Therefore, the overall minimum warm candles required is 14.
MIN_WARM_CANDLES = max(SMA_PERIOD + 1, BB_PERIOD)


def calculate_sma(prices: list[float], period: int) -> float | None:
    """Calculates the Simple Moving Average."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])


def calculate_rsi(closes: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI)."""
    if len(closes) < 2:  # Need at least 2 closes for 1 price change
        return None

    changes = np.diff(closes)  # numpy.diff calculates n-1 differences for n elements

    # Determine the effective period for initial average if data is less than requested period.
    # E.g., if period is 14, but only 13 price changes are available from 14 candles, use 13 for initial average.
    effective_initial_period = min(period, len(changes))

    if effective_initial_period == 0:  # No changes to average
        return None

    gains = np.maximum(0, changes)
    losses = np.maximum(0, -changes)

    # Calculate initial average gain and loss using the effective_initial_period
    avg_gain = np.mean(gains[:effective_initial_period])
    avg_loss = np.mean(losses[:effective_initial_period])

    # Smoothed average gain and loss (RMA) for subsequent periods
    for i in range(effective_initial_period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0  # No losses, RSI is 100
    if avg_gain == 0:
        return 0.0  # No gains, RSI is 0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_band_width_percent(
    closes: list[float], period: int, k: float = BB_K
) -> float | None:
    """Calculates the Bollinger Band Width as a percentage of the Middle Band."""
    if len(closes) < period:
        return None

    period_closes = np.array(closes[-period:])

    middle_band = np.mean(period_closes)
    std_dev = np.std(period_closes)  # Population standard deviation (ddof=0)

    if middle_band == 0:  # Avoid division by zero
        return None

    upper_band = middle_band + (std_dev * k)
    lower_band = middle_band - (std_dev * k)

    bb_width_percent = ((upper_band - lower_band) / middle_band) * 100
    return bb_width_percent


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # --- Data Sufficiency Checks ---
        # 1. Warm data for indicators
        if len(pair_data.warm) < MIN_WARM_CANDLES:
            continue

        # 2. Hot data for current price and spread
        if not pair_data.hot:  # Ensure at least one tick exists
            continue

        current_tick = pair_data.hot[-1]

        # 3. Valid prices in hot data
        if not (
            current_tick.bid_price > 0
            and current_tick.ask_price > 0
            and current_tick.last_price > 0
        ):
            continue

        # 4. Valid previous close for crossover calculation (warm[-2].close)
        # This implicitly checks len(pair_data.warm) >= 2, which is covered by MIN_WARM_CANDLES = 14.
        if not (pair_data.warm[-2].close > 0):
            continue

        # --- Extract required data ---
        current_last_price = current_tick.last_price
        hourly_closes = [candle.close for candle in pair_data.warm]
        previous_hourly_close = pair_data.warm[-2].close  # Close of the candle before the most recent one

        # --- Calculate Indicators ---
        tick_spread_percent: float
        try:
            tick_spread_percent = (
                (current_tick.ask_price - current_tick.bid_price) / current_last_price
            ) * 100
        except ZeroDivisionError:
            # This should ideally be caught by 'current_last_price > 0' check, but as a safeguard.
            continue

        current_sma_9 = calculate_sma(hourly_closes, SMA_PERIOD)
        # previous_sma_9 uses closes up to the second-to-last candle
        previous_sma_9 = calculate_sma(hourly_closes[:-1], SMA_PERIOD)

        current_rsi = calculate_rsi(hourly_closes, RSI_PERIOD)
        current_bb_width_percent = calculate_bollinger_band_width_percent(
            hourly_closes, BB_PERIOD
        )

        # Check if any crucial indicator calculation failed (returned None)
        if any(
            x is None
            for x in [current_sma_9, previous_sma_9, current_rsi, current_bb_width_percent]
        ):
            continue  # Insufficient data or invalid calculation result for indicators

        # --- Entry Conditions (BuySignal) ---
        buy_conditions = [
            # Momentum: current_last_price crosses above current_sma_9
            # Previous hourly close was <= previous SMA, and current tick price is > current SMA
            previous_hourly_close <= previous_sma_9,
            current_last_price > current_sma_9,
            # RSI Filter: Moderate market conditions
            40 <= current_rsi <= 65,
            # Volatility Filter: Non-extreme Bollinger Band Width
            1.0 <= current_bb_width_percent <= 8.0,
            # Liquidity Filter: Low tick spread
            tick_spread_percent < 0.5,
        ]

        if all(buy_conditions):
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=current_tick.polled_at,  # Use tick timestamp for real-time signal
                    price=current_last_price,
                    rule_id=RULE_ID,
                    confidence=1.0,
                )
            )
        else:
            # --- Exit Conditions (SellSignal) ---
            sell_conditions = [
                # Momentum Reversal: current_last_price crosses below current_sma_9
                # Previous hourly close was >= previous SMA, and current tick price is < current SMA
                (previous_hourly_close >= previous_sma_9 and current_last_price < current_sma_9),
                # Overbought Condition: RSI exceeds 65
                (current_rsi > 65),
            ]

            if any(sell_conditions):
                signals.append(
                    SellSignal(
                        pair=pair,
                        timestamp=current_tick.polled_at,
                        price=current_last_price,
                        rule_id=RULE_ID,
                        confidence=1.0,
                    )
                )

    return signals