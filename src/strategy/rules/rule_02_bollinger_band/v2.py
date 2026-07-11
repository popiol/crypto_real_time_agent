from __future__ import annotations

import numpy as np
from datetime import datetime

# Assuming these models are available in the execution environment
# They are not part of this module, but imported for type hinting and data access.
# The `src.agent.models` path is provided in the problem description.
from src.agent.models import (
    BuySignal,
    MarketData,
    PairData,
    SellSignal,
    Tick,
    WarmCandle,
    ColdMonth,
)


# --- Rule Parameters ---
BB_PERIOD = 20
BB_STD_DEV = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70

# Minimum candles required for calculations.
# Bollinger Bands need BB_PERIOD candles.
# RSI needs RSI_PERIOD + 1 candles for the initial calculation.
MIN_CANDLES = max(BB_PERIOD, RSI_PERIOD + 1)

# Unique identifier for this rule, as provided in the idea.
RULE_ID = "bb8cfce1-6b25-4c4f-adaa-42d1d9ad300c"


# --- Helper Functions for Indicator Calculations ---

def _calculate_rsi(prices: np.ndarray, period: int) -> float:
    """
    Calculates the Relative Strength Index (RSI) for the latest price
    in the given array, using Wilder's smoothing method.

    Args:
        prices (np.ndarray): A numpy array of close prices.
        period (int): The RSI period.

    Returns:
        float: The RSI value, or np.nan if insufficient data.
    """
    if len(prices) < period + 1:
        return np.nan

    # Calculate price changes (deltas)
    deltas = np.diff(prices)

    # Separate gains and losses
    gains = np.maximum(0, deltas)
    losses = np.maximum(0, -deltas)  # Losses are positive values

    avg_gain = np.zeros_like(gains)
    avg_loss = np.zeros_like(losses)

    # Calculate the initial average gain/loss over the first 'period' deltas
    # The first 'period' deltas correspond to prices[1]...prices[period+1]
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    # Apply Wilder's smoothing for subsequent periods
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

    # Get the latest average gain and loss
    current_avg_gain = avg_gain[-1]
    current_avg_loss = avg_loss[-1]

    if current_avg_loss == 0:
        if current_avg_gain == 0:
            return 50.0  # No change, neutral RSI
        else:
            return 100.0  # Infinite RS, max RSI (all gains)
    
    rs = current_avg_gain / current_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# --- Main Signal Generation Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on Bollinger Bands with RSI confirmation.

    A Buy signal is generated when the price falls below the lower Bollinger Band
    AND the RSI indicates oversold conditions (RSI < RSI_OVERSOLD_THRESHOLD).

    A Sell signal is generated when the price rises above the upper Bollinger Band
    AND the RSI indicates overbought conditions (RSI > RSI_OVERBOUGHT_THRESHOLD).

    Args:
        data (MarketData): A dictionary of market data for various currency pairs.

    Returns:
        list[BuySignal | SellSignal]: A list of generated buy or sell signals.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for both BB and RSI calculations
        if not pair_data.warm or len(pair_data.warm) < MIN_CANDLES:
            continue

        # Extract close prices from warm candles as a numpy array
        closes = np.array([c.close for c in pair_data.warm])

        # --- Bollinger Band Calculation ---
        # The Bollinger Band calculations use the last BB_PERIOD candles.
        # Since MIN_CANDLES >= BB_PERIOD, we always have enough data.
        bb_prices = closes[-BB_PERIOD:]
        
        sma = np.mean(bb_prices)
        std = np.std(bb_prices)

        # If standard deviation is zero, bands are flat. No meaningful signal.
        if std == 0:
            continue

        upper_band = sma + (std * BB_STD_DEV)
        lower_band = sma - (std * BB_STD_DEV)

        # --- RSI Calculation ---
        # RSI needs at least RSI_PERIOD + 1 prices, which is ensured by MIN_CANDLES.
        rsi_val = _calculate_rsi(closes, RSI_PERIOD)
        
        # If RSI could not be calculated (e.g., due to edge case in helper function
        # or unexpected data pattern, although MIN_CANDLES should prevent this), skip.
        if np.isnan(rsi_val):
            continue

        # --- Get Current Price and Timestamp ---
        # The current price is the last trade price from the most recent tick.
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        ts = current_tick.polled_at

        # --- Generate Signals Based on Rule Logic ---
        # Buy signal: Price below lower band AND RSI is oversold
        if current_price < lower_band and rsi_val < RSI_OVERSOLD_THRESHOLD:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=ts,
                    price=current_price,
                    rule_id=RULE_ID,
                )
            )
        # Sell signal: Price above upper band AND RSI is overbought
        elif current_price > upper_band and rsi_val > RSI_OVERBOUGHT_THRESHOLD:
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=ts,
                    price=current_price,
                    rule_id=RULE_ID,
                )
            )

    return signals