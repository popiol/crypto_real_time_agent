from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# --- Rule Parameters ---
BB_PERIOD = 20
BB_STD_DEV = 2.0
MFI_PERIOD = 14
MFI_OVERSOLD_THRESHOLD = 20
MFI_OVERBOUGHT_THRESHOLD = 80
VOLUME_SMA_PERIOD = 20
VOLUME_MULTIPLIER = 1.5

# Minimum number of candles required for calculations:
# - BB_PERIOD for SMA/STDDEV to have a valid last value.
# - MFI_PERIOD + 2 for MFI to have valid MFI[-1] and MFI[-2] (MFI[period] is the first non-NaN value).
# - VOLUME_SMA_PERIOD for Volume SMA to have a valid last value.
MIN_CANDLES = max(BB_PERIOD, MFI_PERIOD + 2, VOLUME_SMA_PERIOD)


# --- Helper Functions for Indicator Calculations (without TA-Lib) ---

def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < period:
        return np.full_like(data, np.nan)
    sma = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i - period + 1 : i + 1])
    return sma

def _calculate_stddev(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates Standard Deviation."""
    if len(data) < period:
        return np.full_like(data, np.nan)
    stddev = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        stddev[i] = np.std(data[i - period + 1 : i + 1])
    return stddev

def _calculate_bollinger_bands(
    close_prices: np.ndarray, period: int, std_dev_multiplier: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates Bollinger Bands (SMA, Upper, Lower)."""
    sma = _calculate_sma(close_prices, period)
    stddev = _calculate_stddev(close_prices, period)
    upper_band = sma + std_dev_multiplier * stddev
    lower_band = sma - std_dev_multiplier * stddev
    return sma, upper_band, lower_band

def _calculate_mfi(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int
) -> np.ndarray:
    """Calculates Money Flow Index (MFI)."""
    # Need at least period + 1 candles for TP comparisons for the first MFI value
    if len(high) < period + 1:
        return np.full_like(high, np.nan)

    mfi = np.full_like(high, np.nan)
    tp = (high + low + close) / 3.0
    raw_money_flow = tp * volume

    # Calculate initial sums for the first full window (index 'period')
    positive_money_flow_sum = 0.0
    negative_money_flow_sum = 0.0
    for j in range(1, period + 1): # Window from index 1 to period (inclusive)
        if tp[j] > tp[j-1]:
            positive_money_flow_sum += raw_money_flow[j]
        elif tp[j] < tp[j-1]:
            negative_money_flow_sum += raw_money_flow[j]

    # Calculate MFI for the first full window (at index 'period')
    if negative_money_flow_sum == 0:
        mfr = np.inf # Avoid division by zero, MFI will be 100
    else:
        mfr = positive_money_flow_sum / negative_money_flow_sum
    mfi[period] = 100.0 - (100.0 / (1.0 + mfr))

    # Slide the window for subsequent calculations
    for i in range(period + 1, len(high)):
        # Remove oldest bar's contribution from the window
        # The bar to remove is at index `i - period`
        if tp[i - period] > tp[i - period - 1]:
            positive_money_flow_sum -= raw_money_flow[i - period]
        elif tp[i - period] < tp[i - period - 1]:
            negative_money_flow_sum -= raw_money_flow[i - period]

        # Add newest bar's contribution to the window
        # The bar to add is at index `i`
        if tp[i] > tp[i-1]:
            positive_money_flow_sum += raw_money_flow[i]
        elif tp[i] < tp[i-1]:
            negative_money_flow_sum += raw_money_flow[i]

        if negative_money_flow_sum == 0:
            mfr = np.inf
        else:
            mfr = positive_money_flow_sum / negative_money_flow_sum

        mfi[i] = 100.0 - (100.0 / (1.0 + mfr))
    return mfi


# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Identifies Bollinger Band reversal opportunities with MFI turn and volume spike.

    A Buy signal is generated when the price closes below the lower band, MFI is oversold
    and shows an upward turn, and volume is significantly above its recent average.
    A Sell signal is emitted when the price closes above the upper band, MFI is overbought
    and shows a downward turn, with concurrent high volume.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm

        if len(candles) < MIN_CANDLES:
            continue

        # Extract OHLCV data as numpy arrays for calculations
        closes = np.array([c.close for c in candles], dtype=np.float64)
        highs = np.array([c.high for c in candles], dtype=np.float64)
        lows = np.array([c.low for c in candles], dtype=np.float64)
        volumes = np.array([c.volume for c in candles], dtype=np.float64)
        
        # Ensure all arrays are not empty (should be handled by MIN_CANDLES, but defensive check)
        if len(closes) == 0:
            continue

        # Calculate Indicators
        _, upper_band, lower_band = _calculate_bollinger_bands(closes, BB_PERIOD, BB_STD_DEV)
        mfi = _calculate_mfi(highs, lows, closes, volumes, MFI_PERIOD)
        volume_sma = _calculate_sma(volumes, VOLUME_SMA_PERIOD)

        # Check for NaN values at the required indices for the latest signals.
        # This ensures all necessary indicator values are valid.
        if (
            np.isnan(upper_band[-1]) or np.isnan(lower_band[-1]) or
            np.isnan(mfi[-1]) or np.isnan(mfi[-2]) or
            np.isnan(volume_sma[-1])
        ):
            continue

        # Get current and previous values
        current_close = closes[-1]
        current_mfi = mfi[-1]
        previous_mfi = mfi[-2]
        current_volume = volumes[-1]

        last_upper_band = upper_band[-1]
        last_lower_band = lower_band[-1]
        last_volume_sma = volume_sma[-1]

        # Buy Signal Conditions
        if (
            current_close < last_lower_band and # Price breaches lower BB
            current_mfi < MFI_OVERSOLD_THRESHOLD and # MFI is oversold
            current_mfi > previous_mfi and # MFI shows an upward turn from oversold
            current_volume > VOLUME_MULTIPLIER * last_volume_sma # Volume spike
        ):
            signals.append(BuySignal(
                pair=pair,
                timestamp=candles[-1].hour,
                price=current_close,
                rule_id="rule_02_bollinger_band_v6",
                confidence=None
            ))

        # Sell Signal Conditions
        if (
            current_close > last_upper_band and # Price breaches upper BB
            current_mfi > MFI_OVERBOUGHT_THRESHOLD and # MFI is overbought
            current_mfi < previous_mfi and # MFI shows a downward turn from overbought
            current_volume > VOLUME_MULTIPLIER * last_volume_sma # Volume spike
        ):
            signals.append(SellSignal(
                pair=pair,
                timestamp=candles[-1].hour,
                price=current_close,
                rule_id="rule_02_bollinger_band_v6",
                confidence=None
            ))

    return signals