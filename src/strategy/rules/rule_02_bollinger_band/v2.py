"""Rule 02 — Enhanced Bollinger Band with Trend Confirmation."""
from __future__ import annotations

import statistics
from datetime import datetime

# Assuming these models are available in the execution environment
from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick


# --- Rule Parameters ---
PERIOD_BB = 20
STD_DEV_MULTIPLIER = 2.0
PERIOD_SMA_TREND = 50  # New parameter for trend filter

# Minimum candles required for calculations.
# We need enough candles for both BB and SMA_TREND.
MIN_CANDLES_REQUIRED = max(PERIOD_BB, PERIOD_SMA_TREND)

# --- Helper Functions ---
def calculate_sma(prices: list[float]) -> float:
    """Calculates the Simple Moving Average (SMA) for the given list of prices."""
    if not prices:
        raise ValueError("Price list cannot be empty for SMA calculation.")
    return statistics.mean(prices)

def calculate_stddev(prices: list[float]) -> float:
    """Calculates the Standard Deviation for the given list of prices.
    Returns 0.0 if there are fewer than 2 data points (as stdev is undefined or 0).
    """
    if len(prices) < 2:
        return 0.0
    return statistics.stdev(prices)

# --- Main Signal Function ---
def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for all indicator calculations
        if len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue

        # Ensure we have at least one hot tick for current price and timestamp
        if not pair_data.hot:
            continue

        # Extract closing prices from warm candles for indicator calculations
        closes = [c.close for c in pair_data.warm]

        # Calculate Bollinger Bands using the last PERIOD_BB closes
        bb_closes_window = closes[-PERIOD_BB:]
        middle_band = calculate_sma(bb_closes_window)
        std_dev = calculate_stddev(bb_closes_window)

        if std_dev == 0:  # Avoid bands collapsing or division by zero in other contexts
            continue

        upper_band = middle_band + (std_dev * STD_DEV_MULTIPLIER)
        lower_band = middle_band - (std_dev * STD_DEV_MULTIPLIER)

        # Calculate Trend Filter SMA using the last PERIOD_SMA_TREND closes
        sma_trend_closes_window = closes[-PERIOD_SMA_TREND:]
        sma_trend = calculate_sma(sma_trend_closes_window)

        # Get current price and timestamp from the latest hot tick
        current_tick: Tick = pair_data.hot[-1]
        current_price: float = current_tick.last_price
        ts: datetime = current_tick.polled_at

        # Generate signals based on the enhanced rule
        # BUY signal: price below lower BB AND price above long-term SMA (uptrend confirmation)
        if (current_price < lower_band) and (current_price > sma_trend):
            signals.append(BuySignal(
                pair=pair,
                timestamp=ts,
                price=current_price,
                rule_id="dd42a2de-f4e3-43c8-b8f7-e37543bbd2bf" # Unique identifier for this rule idea
            ))
        # SELL signal: price above upper BB AND price below long-term SMA (downtrend confirmation)
        elif (current_price > upper_band) and (current_price < sma_trend):
            signals.append(SellSignal(
                pair=pair,
                timestamp=ts,
                price=current_price,
                rule_id="dd42a2de-f4e3-43c8-b8f7-e37543bbd2bf" # Unique identifier for this rule idea
            ))

    return signals