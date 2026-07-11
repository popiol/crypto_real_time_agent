from __future__ import annotations

import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal

# --- Parameters ---
BB_PERIOD = 20
BB_STD_DEV = 2
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Minimum candles needed for Bollinger Bands: BB_PERIOD
# Minimum candles needed for RSI: RSI_PERIOD + 1 (for initial average gain/loss calculation)
MIN_CANDLES_REQUIRED = max(BB_PERIOD, RSI_PERIOD + 1)

def calculate_rsi(closes: list[float], period: int) -> float:
    """
    Calculates the Relative Strength Index (RSI) for a list of close prices.
    Uses Wilder's smoothing method.

    Args:
        closes: A list of historical close prices.
        period: The lookback period for RSI calculation.

    Returns:
        The RSI value for the last price in the series, or np.nan if insufficient data.
    """
    # Needs at least 'period + 1' prices to calculate 'period' differences for initial average
    if len(closes) < period + 1:
        return np.nan

    closes_arr = np.array(closes, dtype=float)
    diff = np.diff(closes_arr)  # Differences between consecutive closes

    # Separate gains and losses
    gains = np.where(diff > 0, diff, 0)
    losses = np.where(diff < 0, np.abs(diff), 0)

    # Calculate initial average gain and loss over the first 'period' differences
    # These averages correspond to the first 'period' candles after the very first candle in the series.
    # We take a slice of 'period' elements from the 'gains' and 'losses' arrays.
    initial_avg_gain = np.mean(gains[:period])
    initial_avg_loss = np.mean(losses[:period])

    avg_gain = initial_avg_gain
    avg_loss = initial_avg_loss

    # Apply Wilder's smoothing method (EMA-like) for subsequent periods
    # The loop starts from 'period' because the first 'period' diffs were used for the initial average.
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0  # If no losses, RSI is 100 (or 50 if no price change)
    elif avg_gain == 0:
        return 0.0  # If no gains, RSI is 0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates Buy/Sell signals based on Bollinger Band with RSI confirmation.

    A Buy signal is emitted when the price drops below the lower Bollinger Band
    AND the RSI indicates an oversold condition (below RSI_OVERSOLD).

    A Sell signal is emitted when the price rises above the upper Bollinger Band
    AND the RSI indicates an overbought condition (above RSI_OVERBOUGHT).

    Args:
        data: MarketData containing historical and current price information.

    Returns:
        A list of BuySignal or SellSignal objects.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for both BB and RSI calculation
        # and at least one hot tick for current price/timestamp.
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue

        # Use close prices from warm candles for calculations
        closes = [c.close for c in pair_data.warm]

        # --- Bollinger Band Calculation ---
        # We need the last BB_PERIOD closes for SMA and STDDEV
        bb_closes_arr = np.array(closes[-BB_PERIOD:], dtype=float)

        # Skip if there aren't enough closes for BB period (should be caught by MIN_CANDLES_REQUIRED)
        if len(bb_closes_arr) < BB_PERIOD:
            continue
            
        sma = np.mean(bb_closes_arr)
        std = np.std(bb_closes_arr) # Default ddof=0 for population standard deviation

        # If standard deviation is zero, bands collapse to SMA, no meaningful deviation signal
        if std == 0:
            continue

        upper_band = sma + (std * BB_STD_DEV)
        lower_band = sma - (std * BB_STD_DEV)

        # --- RSI Calculation ---
        # The calculate_rsi function needs at least RSI_PERIOD + 1 candles.
        # We pass all available 'closes' to ensure enough history for accurate EMA smoothing,
        # and the function will return the RSI for the *last* candle in the series.
        rsi_value = calculate_rsi(closes, RSI_PERIOD)

        # Handle cases where RSI could not be calculated (e.g., insufficient data, which MIN_CANDLES_REQUIRED tries to prevent)
        if np.isnan(rsi_value):
            continue

        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        ts = current_tick.polled_at

        # --- Signal Generation ---
        # BUY signal: Price below lower BB AND RSI oversold
        if current_price < lower_band and rsi_value < RSI_OVERSOLD:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # SELL signal: Price above upper BB AND RSI overbought
        elif current_price > upper_band and rsi_value > RSI_OVERBOUGHT:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals