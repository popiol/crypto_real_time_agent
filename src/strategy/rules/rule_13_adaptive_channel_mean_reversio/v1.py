from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, Tick

# Parameters
CENTRAL_MA_PERIOD = 20
CHANNEL_WIDTH_VOLATILITY_PERIOD = 10
CHANNEL_WIDTH_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 10
VOLUME_THRESHOLD_MULTIPLIER = 1.5

# Minimum data points required for calculations across all indicators
MIN_DATA_POINTS = max(CENTRAL_MA_PERIOD, CHANNEL_WIDTH_VOLATILITY_PERIOD, VOLUME_MA_PERIOD)

# Helper functions for indicators
def _calculate_ema(series: np.ndarray, period: int) -> np.ndarray:
    """Calculates Exponential Moving Average."""
    if len(series) < period:
        return np.array([])
    alpha = 2 / (period + 1)
    ema_values = np.zeros_like(series, dtype=float)
    # Initialize the first EMA value with an SMA of the first 'period' values
    ema_values[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        ema_values[i] = (series[i] - ema_values[i - 1]) * alpha + ema_values[i - 1]
    return ema_values[period - 1:]

def _calculate_sma(series: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(series) < period:
        return np.array([])
    return np.convolve(series, np.ones(period) / period, mode='valid')

def _calculate_stddev(series: np.ndarray, period: int) -> np.ndarray:
    """Calculates rolling Standard Deviation."""
    if len(series) < period:
        return np.array([])
    # Calculate rolling standard deviation
    std_devs = [np.std(series[i - period:i]) for i in range(period, len(series) + 1)]
    return np.array(std_devs)

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Adaptive Channel Mean Reversion with Volume Confirmation' trading rule.

    This rule identifies mean-reversion opportunities when the price deviates significantly
    from a central moving average within an adaptive channel, confirmed by high volume.
    It uses tick data for both price and a proxy for volume (24h rolling volume).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot # Use tick data for both price and volume consistency

        # Ensure enough data points for all indicator calculations
        if len(ticks) < MIN_DATA_POINTS:
            continue

        # Extract price (last_price) and volume (volume_24h) series from ticks
        # Using last_price as the primary price series
        price_series = np.array([t.last_price for t in ticks], dtype=float)
        # Using volume_24h as a proxy for market activity/volume for confirmation
        volume_series = np.array([t.volume_24h for t in ticks], dtype=float)

        # Re-check data length after conversion to numpy arrays
        if len(price_series) < MIN_DATA_POINTS or len(volume_series) < MIN_DATA_POINTS:
            continue

        # Step 1: Calculate Central Moving Average (EMA)
        central_ma_values = _calculate_ema(price_series, CENTRAL_MA_PERIOD)
        if len(central_ma_values) == 0:
            continue
        central_ma = central_ma_values[-1] # Get the most recent MA value

        # Step 2: Calculate Adaptive Channel Width using Short-term Standard Deviation
        # This replaces ATR as per pseudocode Option B, and is more suitable for tick data
        std_dev_values = _calculate_stddev(price_series, CHANNEL_WIDTH_VOLATILITY_PERIOD)
        if len(std_dev_values) == 0:
            continue
        channel_width = std_dev_values[-1] * CHANNEL_WIDTH_MULTIPLIER

        # Step 3: Define Upper and Lower Bands
        upper_band = central_ma + channel_width
        lower_band = central_ma - channel_width

        # Step 4: Calculate Average Volume (SMA of rolling 24h volume)
        average_volume_values = _calculate_sma(volume_series, VOLUME_MA_PERIOD)
        if len(average_volume_values) == 0:
            continue
        average_volume = average_volume_values[-1] # Get the most recent average volume

        # Step 5: Get Current Price and Volume
        current_price = price_series[-1]
        current_volume = volume_series[-1]
        current_timestamp = ticks[-1].polled_at

        # Step 6: Generate Signals
        # Check for sufficient volume for confirmation
        volume_confirmation_threshold = average_volume * VOLUME_THRESHOLD_MULTIPLIER
        is_high_volume = current_volume > volume_confirmation_threshold

        # Buy Signal: Price touches or crosses below lower band with high volume
        if current_price <= lower_band and is_high_volume:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id="15ae9c16-b00a-4674-9ebc-6df055f2a873",
                confidence=None
            ))
        # Sell Signal: Price touches or crosses above upper band with high volume
        elif current_price >= upper_band and is_high_volume:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id="15ae9c16-b00a-4674-9ebc-6df055f2a873",
                confidence=None
            ))

    return signals