from __future__ import annotations
import statistics
import numpy as np
from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle, Tick
from datetime import datetime

# --- Constants for Bollinger Bands ---
BB_PERIOD = 20  # N-period for SMA and StdDev
BB_K = 2.0      # K multiplier for StdDev

# --- Constants for Stochastic Oscillator ---
STOCH_K_PERIOD = 14     # Period for %K (lowest low, highest high)
STOCH_SLOWING_PERIOD = 3 # Period for smoothing Raw %K to get %K
STOCH_D_PERIOD = 3      # Period for smoothing %K to get %D
STOCH_OVERSOLD = 20.0   # Oversold threshold for %K and %D
STOCH_OVERBOUGHT = 80.0 # Overbought threshold for %K and %D

# Minimum candles required for all calculations
# For Bollinger Bands: BB_PERIOD candles are needed.
# For Stochastic Oscillator:
# To calculate the latest %D, we need STOCH_D_PERIOD smoothed %K values.
# To get STOCH_D_PERIOD smoothed %K values, we need (STOCH_D_PERIOD - 1) + STOCH_SLOWING_PERIOD raw %K values.
# To get (STOCH_D_PERIOD - 1) + STOCH_SLOWING_PERIOD raw %K values, we need
# (STOCH_D_PERIOD - 1) + (STOCH_SLOWING_PERIOD - 1) + STOCH_K_PERIOD total candles.
# This simplifies to STOCH_K_PERIOD + STOCH_SLOWING_PERIOD + STOCH_D_PERIOD - 2.
MIN_CANDLES_REQUIRED = max(BB_PERIOD, STOCH_K_PERIOD + STOCH_SLOWING_PERIOD + STOCH_D_PERIOD - 2)


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average using convolution."""
    if len(data) < period:
        return np.array([])
    return np.convolve(data, np.ones(period)/period, mode='valid')

def _stddev(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Rolling Standard Deviation."""
    if len(data) < period:
        return np.array([])
    result = np.zeros(len(data) - period + 1)
    for i in range(len(result)):
        result[i] = np.std(data[i : i + period])
    return result


def _calculate_stochastic(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    k_period: int,
    slowing_period: int,
    d_period: int
) -> tuple[float | None, float | None]:
    """
    Calculates the current %K and %D values for the Stochastic Oscillator.
    Returns (percent_k, percent_d) or (None, None) if not enough data.
    """
    if len(closes) < k_period:
        return None, None

    # Calculate Raw %K values for all possible windows
    raw_k_values = []
    # Loop from the first possible window (index 0) to the last (index len(closes) - k_period)
    for i in range(len(closes) - k_period + 1):
        window_closes = closes[i : i + k_period]
        window_lows = lows[i : i + k_period]
        window_highs = highs[i : i + k_period]

        highest_high = np.max(window_highs)
        lowest_low = np.min(window_lows)

        if (highest_high - lowest_low) == 0:
            # If high and low are the same, price is flat.
            # Standard practice is often to set %K to 50 in this case.
            raw_k_values.append(50.0)
        else:
            raw_k_values.append(((window_closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100)
    raw_k_values = np.array(raw_k_values)

    # Step 2: Smooth Raw %K to get %K
    if len(raw_k_values) < slowing_period:
        return None, None
    percent_k_values = _sma(raw_k_values, slowing_period)

    # Step 3: Smooth %K to get %D
    if len(percent_k_values) < d_period:
        return None, None
    percent_d_values = _sma(percent_k_values, d_period)

    return percent_k_values[-1], percent_d_values[-1]


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm candles for all calculations
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue

        # Extract data for calculations
        # We use closes from warm candles for BB and Stochastic,
        # and the latest hot tick price for the current price check.
        closes = np.array([c.close for c in pair_data.warm])
        highs = np.array([c.high for c in pair_data.warm])
        lows = np.array([c.low for c in pair_data.warm])
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # --- Bollinger Band Calculation ---
        # MIN_CANDLES_REQUIRED ensures len(closes) is at least BB_PERIOD,
        # so _sma and _stddev will always return non-empty arrays here.
        mb_values = _sma(closes, BB_PERIOD)
        middle_band = mb_values[-1]

        std_dev_values = _stddev(closes, BB_PERIOD)
        std_dev = std_dev_values[-1]

        if std_dev == 0:
            # If standard deviation is zero, price is flat, bands would be on top of MB.
            # No meaningful deviation, so no signal.
            continue

        upper_band = middle_band + (BB_K * std_dev)
        lower_band = middle_band - (BB_K * std_dev)

        # --- Stochastic Oscillator Calculation ---
        percent_k, percent_d = _calculate_stochastic(
            highs, lows, closes,
            STOCH_K_PERIOD, STOCH_SLOWING_PERIOD, STOCH_D_PERIOD
        )

        if percent_k is None or percent_d is None:
            # This check is mostly redundant due to MIN_CANDLES_REQUIRED,
            # but serves as defensive programming for the helper function.
            continue

        # --- Signal Logic ---
        # Buy Signal: Price below Lower Band AND Stochastic indicates oversold conditions
        if (current_price < lower_band and
                percent_k < STOCH_OVERSOLD and
                percent_d < STOCH_OVERSOLD):
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
        # Sell Signal: Price above Upper Band AND Stochastic indicates overbought conditions
        elif (current_price > upper_band and
                percent_k > STOCH_OVERBOUGHT and
                percent_d > STOCH_OVERBOUGHT):
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))

    return signals